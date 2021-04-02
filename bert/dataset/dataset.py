# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:28
import os
import tqdm
import torch
import pickle
import random
import numpy as np
import scipy.sparse as sp
from typing import Dict, List
from collections import OrderedDict
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import networkx as nx
from .masking import MolSpanMaskingScheme


class MolBertDataset(Dataset):
    # add option 'adj_path' and 'with_adj'
    def __init__(self, corpus_path, vocab, seq_len,
                 with_span=False, with_mask=False, with_proximity=False, adj_path=None,
                 geometric_p=0.2, span_lower=1, span_upper=5,
                 encoding="utf-8", corpus_lines=None, on_memory=True):

        assert ((with_span or with_mask or with_proximity) and adj_path is not None) \
               or (not (with_span or with_mask or with_proximity) and adj_path is None)

        self.corpus_path = corpus_path
        self.vocab = vocab
        self.seq_len = seq_len
        self.with_span = with_span
        self.with_mask = with_mask
        self.with_proximity = with_proximity
        self.adj_path = adj_path
        self.geometric_p = geometric_p
        self.span_lower = span_lower
        self.span_upper = span_upper

        self.encoding = encoding
        self.corpus_lines = corpus_lines
        self.on_memory = on_memory

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                # -1 in order to strip '\n'
                # self.lines = [line[:-1].split("\t")
                #               for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.lines = [line[:-1]
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

        if adj_path is not None:
            if os.path.getsize(adj_path) > 0:
                with open(adj_path, "rb") as reader:
                    unpickler = pickle.Unpickler(reader)
                    self.adjs = unpickler.load()
            else:
                print('Empty file...')
        else:
            self.adjs = None

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t = self.random_sent(item)
        if self.adjs is not None:
            adj = self.adjs[item]
            adj = adj.todense()
            if not self.with_span:
                t_gen, t_label, masked_adj, candidate_mask = self.gen_word(t, adj)
            else:
                t_random, t_label, masked_adj = self.span_mask_word(t, adj, self.vocab)
        else:
            t_gen, t_label, candidate_mask = self.gen_word(t)

        # add 'sos_index' and 'eos_index‘
        # mask operation for masked language model
        t = [self.vocab.sos_index] + t_gen + [self.vocab.eos_index]
        t_label = [self.vocab.sos_index] + t_label + [self.vocab.eos_index]
        candidate_mask = [0] + candidate_mask + [0]

        # truncate
        segment_label = ([1 for _ in range(len(t))])[:self.seq_len]
        # print(t, t_label, segment_label)
        bert_input = t[:self.seq_len]
        bert_label = t_label[:self.seq_len]
        bert_mask = candidate_mask[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        padding_mask = [0 for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
        bert_mask.extend(padding_mask)

        output = {
            "bert_input": bert_input,
            "bert_label": bert_label,
            "segment_label": segment_label,
            "bert_mask": bert_mask,
        }

        if self.with_mask or self.with_proximity or self.with_span:
            # adj = self.adjs[item]
            # version 1
            # adj = self.process_adj(adj)
            # version 2
            # adj = self.process_adj_v2(adj)
            # truncation operation for consistent length
            masked_adj = self.process_adj_v2(masked_adj)
            bert_adj = self.process_adj_v2(adj)
            output.update({"masked_adj": masked_adj})
            output.update({"bert_adj": bert_adj})

        return {key: torch.tensor(value) for key, value in output.items()}

    def gen_word(self, sentence, adj=None):
        tokens = sentence.split()
        output_label = []

        masked_indices = []
        candidate_mask = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index
                    masked_indices.append(i)
                    # masked_lm_labels.append(i)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))
                    masked_indices.append(i)
                    # masked_lm_labels.append(i)

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                candidate_mask.append(0)

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)
                candidate_mask.append(1)
        # print(output_label)
        if adj is not None:
            for index in masked_indices:
                # row
                adj[index, :] = 0
                # col
                adj[:, index] = 0

            return tokens, output_label, adj, candidate_mask

        return tokens, output_label, candidate_mask

    def span_mask_word(self, sentence, adj, vocab):
        tokens = sentence.split(' ')
        scheme = MolSpanMaskingScheme(self.geometric_p, 1, int(len(tokens) * 0.2))
        tokens, output_label, masked_adj = scheme.mask(tokens, adj, vocab)
        return tokens, output_label, masked_adj

    def random_sent(self, index):
        t = self.get_corpus_line(index)
        return t

    def get_corpus_line(self, item):

        if self.on_memory:
            return self.lines[item]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t = line[:-1]
            return t

    def process_adj(self, adj: Dict[str, List[str]]) -> np.ndarray:
        """
        relative position matters.
        :param adj:
        :return:
        """
        fps = list(adj.keys())
        num_fps = len(fps)
        fp_adj = np.zeros((self.seq_len, self.seq_len), dtype=np.float32)
        # account for sos_index, we must '+1'
        fp2idx = {fp: idx + 1 for idx, fp in enumerate(fps)}
        # processing 'sos_index'
        # first row
        fp_adj[0, :(num_fps + 2)] = 1.0
        # first column
        fp_adj[:(num_fps + 2), 0] = 1.0
        # processing 'eos_index'
        # last row
        fp_adj[(num_fps + 1), :(num_fps + 2)] = 1.0
        # last column
        fp_adj[:(num_fps + 2), (num_fps + 1)] = 1.0
        # process indices except 'sos_index' and 'eos_index'
        for fp in fps:
            idx = fp2idx[fp]
            # self-loop
            fp_adj[idx, idx] = 1.0
            adj_fps = adj[fp]
            adj_indices = [fp2idx[adj_fp] for adj_fp in adj_fps]
            for adj_idx in adj_indices:
                fp_adj[idx, adj_idx] = 1.0
                fp_adj[adj_idx, idx] = 1.0
        return fp_adj

    def process_adj_v2(self, adj: sp.coo_matrix) -> np.ndarray:
        """
        relative position matters.
        :param adj:
        :return:
        """
        if isinstance(adj, sp.coo_matrix):
            adj = adj.todense()
        num_fps = adj.shape[0]
        fp_adj = np.zeros((self.seq_len, self.seq_len), dtype=np.float32)
        # processing 'sos_index'
        # first row
        fp_adj[0, :(num_fps + 2)] = 1.0
        # first column
        fp_adj[:(num_fps + 2), 0] = 1.0
        # processing 'eos_index'
        # last row
        fp_adj[(num_fps + 1), :(num_fps + 2)] = 1.0
        # last column
        fp_adj[:(num_fps + 2), (num_fps + 1)] = 1.0
        # process indices except 'sos_index' and 'eos_index'
        fp_adj[1: (1 + num_fps), 1: (1 + num_fps)] = adj

        return fp_adj

    # def convert_tokens_to_ids(self, tokens):
    #     """Converts a sequence of tokens into ids using the vocab."""
    #     ids = []
    #     for token in tokens:
    #         ids.append(self.vocab[token])
    #     if len(ids) > self.seq_len:
    #         logger.warning(
    #             "Token indices sequence length is longer than the specified maximum "
    #             " sequence length for this BERT model ({} > {}). Running this"
    #             " sequence through BERT will result in indexing errors".format(len(ids), self.seq_len)
    #         )
    #     return ids