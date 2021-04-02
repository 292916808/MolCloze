# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:29

import random
import numpy as np
import networkx as nx
from bert.walks import DeepWalker, Node2VecWalker, BFSWalker, DFSWalker


class MolSpanMaskingScheme(object):

    def __init__(self, geometric_p, span_lower, span_upper):
        self.geometric_p = geometric_p
        self.span_lower = span_lower
        self.span_upper = span_upper
        self.lens = list(range(span_lower, span_upper + 1))
        len_distrib = [self.geometric_p * (1 - self.geometric_p) ** (i - self.span_lower)
                       for i in range(self.span_lower, self.span_upper + 1)] if self.geometric_p >= 0 else None
        self.len_distrib = [x / (sum(len_distrib)) for x in len_distrib]

    def mask(self, tokens, adj, vocab):
        graph = nx.from_numpy_array(adj)
        assert len(tokens) == len(graph)
        # walker = DeepWalker(graph)
        # walker = BFSWalker(graph)
        walker = DFSWalker(graph)

        num_total_fps = len(tokens)
        num_masked_fps = int(num_total_fps * 0.15 * 0.8)
        num_random_fps = int(num_total_fps * 0.15 * 0.1)
        num_unchanged_fps = int(num_total_fps * 0.15 * 0.1)
        candidate_start_nodes = np.random.choice(graph.nodes(), size=num_total_fps, replace=False)

        start_node = candidate_start_nodes[0]
        masked_walk = walker.walk(walk_length=num_masked_fps, start_node=start_node)
        start_node = candidate_start_nodes[1]
        random_walk = walker.walk(walk_length=num_random_fps, start_node=start_node)
        start_node = candidate_start_nodes[2]
        unchanged_walk = walker.walk(walk_length=num_unchanged_fps, start_node=start_node)

        # start_node = candidate_start_nodes[0]
        # walks = walker.walk(num_masked_fps + num_random_fps + num_unchanged_fps, start_node)

        output_label = []
        masked_indices = []
        for i, token in enumerate(tokens):
            # if i in walks:
            #     prob = random.random()
            #     # 80% randomly change token to mask token
            #     if prob < 0.8:
            #         tokens[i] = vocab.mask_index
            #     # 10% randomly change token to random token
            #     elif prob < 0.9:
            #         tokens[i] = random.randrange(len(vocab))
            #     # 10% randomly change token to current token
            #     else:
            #         tokens[i] = vocab.stoi.get(token, vocab.unk_index)
            #     output_label.append(vocab.stoi.get(token, vocab.unk_index))
            if (i in masked_walk) or (i in random_walk) or (i in unchanged_walk):
                if i in masked_walk:
                    tokens[i] = vocab.mask_index
                    masked_indices.append(i)
                elif i in random_walk:
                    tokens[i] = random.randrange(len(vocab))
                    masked_indices.append(i)
                elif i in unchanged_walk:
                    tokens[i] = vocab.stoi.get(token, vocab.unk_index)
                output_label.append(vocab.stoi.get(token, vocab.unk_index))
            else:
                tokens[i] = vocab.stoi.get(token, vocab.unk_index)
                output_label.append(0)

            for index in masked_indices:
                # row
                adj[index, :] = 0
                # col
                adj[:, index] = 0

        return tokens, output_label, adj
