# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:41
import sys
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .optim_schedule import Scheduleoptim
from bert.model import MolBertLM, MolBertDis, LinearActivation
from bert.model.decoder import InnerProductDecoder, BilinearDecoder, NNBilinearDecoder

def sampling(input_ids, generator_logits, masked_lm_labels):
    generator_id = torch.argmax(generator_logits, dim=-1).detach()
    origin_input = input_ids.clone()
    fake_input = torch.where(masked_lm_labels < 1, origin_input, generator_id)
    corrupt_label = (masked_lm_labels != 0)
    origin_input[corrupt_label] = masked_lm_labels[corrupt_label]
    discriminator_label = torch.eq(origin_input, fake_input)
    return generator_id, discriminator_label

def get_energy_loss(device, logits, targets, weights, vocab_size, batch_size, seq_len=140):
    # targets = torch.unsqueeze(targets, dim=1)
    # print(torch.zeros(batch_size, seq_len, vocab_size).size(), targets.unsqueeze(-1).size())
    oh_labels = torch.zeros(batch_size, seq_len, vocab_size).to(device).scatter_(2, targets.unsqueeze(-1), 1)
    # ones = torch.sparse.torch.eye(vocab_size).to(device)
    # oh_labels = ones.index_select(0, targets)
    log_probs = F.log_softmax(logits, dim=-1)
    # print(log_probs.size(), oh_labels.size())
    label_log_probs = -torch.sum(log_probs * oh_labels, dim=-1)
    # print(weights.is_cuda, label_log_probs.is_cuda)
    numerator = torch.sum(weights * label_log_probs)
    denominator = torch.sum(weights) + 1e-6
    loss = numerator/denominator
    return loss

def get_token_energy_logits(device, LinearAct, inputs, table, weights, targets, vocab_size, batch_size):
    energy_hidden = LinearAct(inputs)
    # print(energy_hidden.size(), table.size())
    logits = torch.matmul(energy_hidden, table.transpose(-2,-1))
    energy_loss = get_energy_loss(device, logits, targets, weights, vocab_size, batch_size)
    return logits, energy_loss

def get_cloze_outputs(device, candidate_mask, LinearAct, inputs, table, targets, vocab_size, batch_size):
    weights = candidate_mask.type(torch.FloatTensor).to(device)
    logits, energy_loss = get_token_energy_logits(device, LinearAct, inputs, table, weights, targets, vocab_size, batch_size)
    return logits, energy_loss, weights

def get_discriminator_energy_loss(device, hidden, dis_output, criterion, input_ids, cloze_logits, weights, vocab_size, \
                                    batch_size, discriminator_label, candidate_mask, mask_prob=0.15, seq_len=140):
    # print(input_ids.size())
    oh_labels = torch.zeros(batch_size, seq_len, vocab_size).to(device).scatter_(2, input_ids.unsqueeze(-1), 1)   
    # print(cloze_logits.size(), oh_labels.size())
    log_q = torch.sum(F.log_softmax(cloze_logits, -1) * oh_labels.type(torch.FloatTensor).to(device), dim=-1).detach() 
    # print(logits.size(), log_q.size())
    logits = torch.squeeze(hidden(dis_output), -1)
    logits += log_q 
    logits += torch.log(mask_prob/(1-mask_prob)*torch.ones_like(logits))
    unmask_labels = torch.mul(discriminator_label, candidate_mask).type(torch.FloatTensor).to(device)
    losses = criterion(logits, unmask_labels) * weights
    loss = torch.sum(losses)/(torch.sum(weights)+1e-6)
    return loss

class MolBertTrainer(object):

    def __init__(self, bert, vocab_size, 
                 train_loader, test_loader, batch_size,
                 lr=1e-4, betas=(0.9, 0.999),
                 weight_decay=0.01, warmup_steps=10000,
                 with_cuda=True, cuda_devices=None,
                 with_span=False, with_mask=False, with_proximity=False,
                 log_freq=10, logger=None):

        super(MolBertTrainer, self).__init__()
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.vocab_size = vocab_size
        self.bert = bert
        self.model = MolBertLM(bert, vocab_size).to(self.device)
        self.discriminator = MolBertDis(bert).to(self.device)
        self.get_energy_logits = LinearActivation(self.bert.hidden, self.bert.hidden).to(self.device)
        self.dis_energy_preds = LinearActivation(self.bert.hidden, 1).to(self.device)

        if with_cuda and torch.cuda.device_count() > 1:
            logger.info('Using {} GPUs for Bert'.format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
            self.discriminator = nn.DataParallel(self.discriminator, device_ids=cuda_devices)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = Scheduleoptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps, grad_clip=5.)

        self.criterion = nn.NLLLoss(ignore_index=0)
        self.dis_det_criterion = nn.BCEWithLogitsLoss()
        self.dis_ene_criterion = nn.BCEWithLogitsLoss()

        self.with_span = with_span
        self.with_mask = with_mask
        self.with_proximity = with_proximity
        if with_proximity:
            # self.decoder = BilinearDecoder(input_dim=bert.hidden)
            self.decoder = InnerProductDecoder(input_dim=bert.hidden)
            self.dec_criterion = nn.MSELoss()
        self.log_freq = log_freq
        self.logger = logger
        self.batch_size = batch_size

        logger.info('Total parameters: {}'.format(
            sum([p.nelement() for p in self.model.parameters()])+sum([p.nelement() for p in self.discriminator.parameters()])
        ))

    # def _tie_embeddings(self):

    def train(self, epoch):
        self.iteration(epoch, self.train_loader)

    def test(self, epoch):
        self.iteration(epoch, self.test_loader, train=False)

    def iteration(self, epoch, loader, train=True):
        logger = self.logger
        str_code = 'train' if train else 'test'
        data_iter = tqdm.tqdm(
            enumerate(loader),
            desc='{} : epoch {}'.format(str_code, epoch),
            total=len(loader),
            bar_format='{l_bar}{r_bar}'
        )

        avg_loss = 0.0
        avg_dec_loss = 0.0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device
            data = {key: value.to(self.device) for key, value in data.items()}
            bert_table = [i for i in range(self.vocab_size)]
            bert_table = torch.LongTensor(bert_table).to(self.device)
            bert_table = self.bert.embedding.token(bert_table)
            # 1. forward masked_lm_model
            # # for span masking
            # mask_lm_output, bert_outputs = self.model.forward(data['bert_input'], data['segment_label'])
            if self.with_span:
                mask_lm_output, _, bert_output = self.model.forward(data['bert_input'], data['segment_label'],
                                                                  data['masked_adj'])
            elif self.with_mask:
                mask_lm_output, _, bert_output = self.model.forward(data['bert_input'], data['segment_label'],
                                                                  data['masked_adj'])
            elif self.with_proximity:
                mask_lm_output, _, bert_output = self.model.forward(data['bert_input'], data['segment_label'],
                                                                  data['masked_adj'])
                rec_adj = self.decoder(bert_outputs[-1])
                ori_adj = data['bert_adj'].view(-1)
                dec_loss = self.dec_criterion(rec_adj, ori_adj)
            else:
                mask_lm_output, _, bert_output = self.model.forward(data['bert_input'], data['segment_label'])

            # print(data['bert_input'].size()[0])
            # 2-1. NLL(negative log likelihood) loss
            # mask_lm_output = (mask_lm_output < 1e-6) * torch.ones_like(mask_lm_output) * 1e-6 + (mask_lm_output >= 1e-6) * mask_lm_output
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data['bert_label'])
            # print(data['bert_table'][0])
            # print(self.bert.embedding.token(data['bert_table'][0]).size())
            cloze_logits, gen_ene_loss, weights = get_cloze_outputs(self.device, data['bert_mask'], self.get_energy_logits, bert_output, \
                                                                        bert_table, data['bert_input'], self.vocab_size, data['bert_input'].size()[0])
            discriminator_input, discriminator_label = sampling(data['bert_input'], bert_output, data['bert_label'])
            if self.with_mask:
                dis_preds, dis_output = self.discriminator.forward(discriminator_input, data['segment_label'], data['masked_adj'])
            else:
                dis_preds, dis_output = self.discriminator.forward(discriminator_input, data['segment_label'])
        
            # print(dis_output.size())
            # print(data['bert_label'].size(), mask_lm_output.size(), dis_output.size())
            dis_det_loss = self.dis_det_criterion(dis_preds.view(-1), discriminator_label.view(-1).float())
            dis_ene_loss = get_discriminator_energy_loss(self.device, self.dis_energy_preds, dis_output, self.dis_ene_criterion, discriminator_input, cloze_logits, weights, \
                                                            self.vocab_size, self.batch_size, discriminator_label, data['bert_mask'])
            # print(gen_ene_loss, dis_ene_loss)
            loss = mask_loss + 20 * dis_det_loss + (gen_ene_loss + dis_ene_loss)

            if np.isnan(loss.item()):
                print(data['bert_input'])
                print(mask_lm_output)
                sys.exit(-1)

            if self.with_proximity:
                loss += dec_loss
                avg_dec_loss += dec_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            avg_loss += loss.item()

            post_fix = {
                'epoch': epoch,
                'iter': i,
                'avg_loss': avg_loss / (i + 1),
                'loss': loss.item(),
                'mask_loss': mask_loss.item(),
                'gen_ene_loss': gen_ene_loss.item(),
                'dis_det_loss': dis_det_loss.item(),
                'dis_ene_loss': dis_ene_loss.item()
            }
            if self.with_proximity:
                post_fix.update(
                    {'avg_dec_loss': avg_dec_loss.item() / (i + 1)}
                )

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        logger.info('{} : epoch {}, avg_loss = {:.4f}'.format(
            str_code, epoch, avg_loss / len(data_iter)
        ))

    def save(self, epoch, file_path='output/bert_trained.model'):
        logger = self.logger
        output_filepath = file_path + '.ep{}'.format(epoch)
        torch.save(self.bert.cpu(), output_filepath)
        self.bert.to(self.device)
        logger.info('Epoch {:>3d} Model save on: {}'.format(
            epoch, output_filepath
        ))
