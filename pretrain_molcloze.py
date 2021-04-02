# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:49
import os
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace

from bert.utils.logger_utils import create_logger
from bert.utils.data_utils import get_pubchem_zinc_path, get_available_data_types
from bert.dataset import WordVocab, MolBertDataset
from bert.model import MolBert
from bert.training import MolBertTrainer

pubchem_zinc_path = get_pubchem_zinc_path()


def add_args():
    parser = ArgumentParser()
    # pretrain dataset
    parser.add_argument('--data_type', type=str, choices=get_available_data_types() + ['baai'], default='1m')
    parser.add_argument('--suffix', type=str, choices=['.txt', '.smi'], default='.txt')
    parser.add_argument('--min_freq', type=int, default=5)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--train_dataset', type=str, default=None)
    parser.add_argument('--test_dataset', type=str, default=None)
    parser.add_argument('--vocab_path', type=str, default=None)
    # parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--on_memory', type=bool, default=True)
    parser.add_argument('--corpus_lines', type=int, default=None)

    # bert architecture
    parser.add_argument('--hidden', type=int, default=768)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--attn_heads', type=int, default=12)
    parser.add_argument('--seq_len', type=int, default=140)
    parser.add_argument('--with_span', action='store_true', default=False)
    parser.add_argument('--with_mask', action='store_true', default=False)
    parser.add_argument('--with_proximity', action='store_true', default=False)
    parser.add_argument('--residual_type', type=str, choices=['graph_raw', 'graph_naive', 'raw', 'naive'],
                        default='naive')
    parser.add_argument('--without_ffn', action='store_true', default=False,
                        help='whether not to use ffn in transformer.')

    # training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam_weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)

    # device
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--cuda_devices', type=int, nargs='+', default=None)

    # output
    parser.add_argument('--log_freq', type=int, default=10)

    args = parser.parse_args()
    return args


def modify_args(args: Namespace):
    args.with_adj = args.with_span or args.with_mask or args.with_proximity
    if args.data_type in get_available_data_types():
        if not args.with_adj:
            if args.suffix == '.smi':
                args.train_dataset = os.path.join(
                    pubchem_zinc_path, 'train_{}'.format(args.data_type), 'total.smi'
                )
                args.test_dataset = os.path.join(
                    pubchem_zinc_path, 'valid_{}'.format(args.data_type), 'total.smi'
                )
                args.vocab_path = os.path.join(
                    pubchem_zinc_path, 'train_{}'.format(args.data_type), 'bert_vocab_min{}.pkl'.format(args.min_freq)
                )
            elif args.suffix == '.txt':
                args.train_dataset = os.path.join(
                    pubchem_zinc_path, 'train_{}'.format(args.data_type),
                    'sentences.txt' if args.radius == 1 else 'sentences_r{}.txt'.format(args.radius)
                )
                args.test_dataset = os.path.join(
                    pubchem_zinc_path, 'valid_{}'.format(args.data_type),
                    'sentences.txt' if args.radius == 1 else 'sentences_r{}.txt'.format(args.radius)
                )
                args.vocab_path = os.path.join(
                    pubchem_zinc_path, 'train_{}'.format(args.data_type),
                    'bert_vocab_min{}_txt.pkl'.format(args.min_freq) if args.radius == 1 else 'bert_vocab_min{}_r{}_txt.pkl'.format(args.min_freq, args.radius)
                )
        else:
            if args.suffix == '.txt':
                args.train_dataset = os.path.join(
                    pubchem_zinc_path, 'train_{}'.format(args.data_type),
                    'sentences.txt' if args.radius == 1 else 'sentences_r{}.txt'.format(args.radius)
                )
                args.train_adj_filepath = os.path.join(
                    pubchem_zinc_path, 'train_{}'.format(args.data_type),
                    'adjs.pkl' if args.radius == 1 else 'adjs_r{}.pkl'.format(args.radius)
                )
                args.test_dataset = os.path.join(
                    pubchem_zinc_path, 'valid_{}'.format(args.data_type),
                    'sentences.txt' if args.radius == 1 else 'sentences_r{}.txt'.format(args.radius)
                )
                args.test_adj_filepath = os.path.join(
                    pubchem_zinc_path, 'valid_{}'.format(args.data_type),
                    'adjs.pkl' if args.radius == 1 else 'adjs_r{}.pkl'.format(args.radius)
                )
                args.vocab_path = os.path.join(
                    pubchem_zinc_path, 'train_{}'.format(args.data_type),
                    'bert_vocab_min{}_txt.pkl'.format(args.min_freq) if args.radius == 1 else 'bert_vocab_min{}_r{}_txt.pkl'.format(args.min_freq, args.radius)
                )
            else:
                raise ValueError('Suffix must be .txt when using adj.')
    if not args.with_adj:
        if args.suffix == '.smi':
            args.output_path = 'output_bert_{}_min{}_H{}_L{}_A{}'.format(
                args.data_type, args.min_freq,
                args.hidden, args.layers, args.attn_heads)
        elif args.suffix == '.txt':
            args.output_path = 'output_bert_{}_min{}_H{}_L{}_A{}_txt'.format(
                args.data_type, args.min_freq,
                args.hidden, args.layers, args.attn_heads)
            if args.radius != 1:
                args.output_path = 'output_bert_{}_min{}_r{}_H{}_L{}_A{}_txt'.format(
                    args.data_type, args.min_freq, args.radius,
                    args.hidden, args.layers, args.attn_heads)
        else:
            raise ValueError('No such suffix named {}.'.format(args.suffix))
    else:
        if args.suffix == '.txt':
            if args.with_span: adj_strategy = 'span'
            elif args.with_mask: adj_strategy = 'mask'
            elif args.with_proximity: adj_strategy = 'prox'
            else: adj_strategy = 'none'
            args.output_path = 'output_bert_{}_min{}_H{}_L{}_A{}_{}_txt'.format(
                args.data_type, args.min_freq,
                args.hidden, args.layers, args.attn_heads, adj_strategy
            )
            if args.radius != 1:
                args.output_path = 'output_bert_{}_min{}_r{}_H{}_L{}_A{}_{}_txt'.format(
                    args.data_type, args.min_freq, args.radius,
                    args.hidden, args.layers, args.attn_heads, adj_strategy)
            if args.residual_type not in ['none', 'graph_raw']:
                args.output_path = 'output_bert_{}_min{}_H{}_L{}_A{}_{}_{}_txt'.format(
                    args.data_type, args.min_freq,
                    args.hidden, args.layers, args.attn_heads, adj_strategy, args.residual_type
                )
            if args.without_ffn:
                args.output_path = args.output_path + '_{}'.format('wo_ffn')
        else:
            raise ValueError('Suffix must be .txt when using adj.')

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    args.with_cuda = True if args.gpu is not None else False
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def parse_args():
    args = add_args()
    modify_args(args)
    return args


def train():
    logger.info('Loading vocab from {}'.format(args.vocab_path))
    vocab = WordVocab.load_vocab(args.vocab_path)

    logger.info('Vocab size: {}'.format(len(vocab)))
    logger.info('Output path: {}'.format(args.output_path))

    logger.info('Loading train dataset from {}'.format(args.train_dataset))
    train_dataset = MolBertDataset(
        args.train_dataset, vocab,
        seq_len=args.seq_len,
        corpus_lines=args.corpus_lines,
        on_memory=args.on_memory,
        with_span=args.with_span,
        with_mask=args.with_mask,
        with_proximity=args.with_proximity,
        adj_path=args.train_adj_filepath if args.with_adj else None,
    )
    logger.info('Loading test dataset from {}'.format(args.test_dataset))
    test_dataset = MolBertDataset(
        args.test_dataset, vocab,
        seq_len=args.seq_len,
        corpus_lines=args.corpus_lines,
        on_memory=args.on_memory,
        with_span=args.with_span,
        with_mask=args.with_mask,
        with_proximity=args.with_proximity,
        adj_path=args.test_adj_filepath if args.with_adj else None,
    )

    logger.info('Creating dataloader')
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    logger.info('Building Bert model')
    bert = MolBert(len(vocab), hidden=args.hidden,
                      n_layers=args.layers,
                      attn_heads=args.attn_heads,
                      residual_type=args.residual_type,
                      without_ffn=args.without_ffn,
                      )

    logger.info('Creating Bert trainer')
    trainer = MolBertTrainer(
        bert, len(vocab), train_loader, test_loader, batch_size=args.batch_size,
        lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        with_span=args.with_span,
        with_mask=args.with_mask,
        with_proximity=args.with_proximity,
        with_cuda=args.with_cuda, cuda_devices=args.cuda_devices,
        log_freq=args.log_freq, logger=logger
    )
    logger.info('Training start')
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, os.path.join(args.output_path, 'model.pt'))

        if test_loader is not None:
            trainer.test(epoch)


args = parse_args()
logger = create_logger(__file__.split('.')[0], save_dir=args.output_path)

if __name__ == '__main__':
    train()
