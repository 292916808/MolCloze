# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:35
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        """
        :param query: (batch_size, attn_heads, max_atoms, d_k)
        :param key: (batch_size, attn_heads, max_atoms, d_k)
        :param value: (batch_size, attn_heads, max_atoms, d_k)
        :param mask: (batch_size, 1, max_atoms, max_atoms)
        :param dropout:
        :return:
        """
        # scores: (batch_size, attn_heads, max_atoms, d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class EdgeGuidedAttention(nn.Module):

    def __init__(self, d_k):
        super(EdgeGuidedAttention, self).__init__()
        self.W_attn_h = nn.Linear(d_k * 2, d_k)
        self.W_attn_o = nn.Linear(d_k, 1)

    def forward(self, query, key, value, mask=None, dropout=None):
        """
        :param query: (batch_size, attn_heads, max_atoms, d_k)
        :param key:  (batch_size, attn_heads, max_atoms, d_k)
        :param value: (batch_size, attn_heads, max_atoms, d_k)
        :param mask: (batch_size, 1, max_atoms, max_atoms)
        :param dropout:
        :return:
        """
        batch_size, attn_heads, max_atoms, _ = query.size()
        # attn_input: (batch_size * attn_heads, max_atoms, max_atoms, 2 * d_k)
        attn_input = self.compute_attn_input(query, key)
        # attn_scores: (batch_size * attn_heads, max_atoms, max_atoms, 1)
        attn_scores = nn.LeakyReLU(0.2)(
            self.W_attn_h(attn_input)
        )
        attn_scores = self.W_attn_o(attn_scores)
        attn_scores = attn_scores.squeeze(-1).view(batch_size, attn_heads, max_atoms, max_atoms)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # p_attn: (batch_size, attn_heads, max_atoms, max_atoms)
        p_attn = F.softmax(attn_scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        # value: (batch_size, attn_heads, max_atoms, d_k)
        return torch.matmul(p_attn, value), p_attn

    def compute_attn_input(self, query, key):
        """
        :param query: (batch_size, attn_heads, max_atoms, d_k), centering node
        :param key: (batch_size, attn_heads, max_atoms, d_k), neighboring nodes
        :return:
        """
        # query: (batch_size * attn_heads, max_atoms, d_k)
        batch_size, attn_heads, max_atoms, d_k = query.size()
        query = query.reshape(batch_size * attn_heads, max_atoms, d_k)
        # key: (batch_size * attn_heads, max_atoms, d_k)
        key = key.reshape(batch_size * attn_heads, max_atoms, d_k)

        # attn_input is concatenation of node pair embeddings
        # query: (batch_size * attn_heads, max_atoms, max_atoms, d_k)
        query = query.unsqueeze(2).expand(-1, -1, max_atoms, -1)
        # key: (batch_size * attn_heads, max_atoms, max_atoms, d_k)
        key = key.unsqueeze(1).expand(-1, max_atoms, -1, -1)

        # attn_input: (batch_size * attn_heads, max_atoms, max_atoms, 2 * d_k)
        attn_input = torch.cat([query, key], dim=3)

        return attn_input

