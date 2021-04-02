# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:40
import torch
import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, raw_x=None, adj=None):
        return x + self.dropout(sublayer(self.norm(x)))
        # the statement above may cause nan error
        # return self.norm(x + self.dropout(sublayer(x)))


class RawSublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(RawSublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, raw_x=None, adj=None, update_layer=None):
        transformed_x = raw_x

        if update_layer is not None:
            hidden = transformed_x
            input = self.dropout(sublayer(self.norm(x)))
            mb, seq_len, hidden_size = hidden.size()
            hidden = hidden.view(mb * seq_len, hidden_size)
            input = input.view(mb * seq_len, hidden_size)
            new_hidden = update_layer(input, hidden)
            new_hidden = new_hidden.view(mb, seq_len, hidden_size)
            return new_hidden

        return transformed_x + self.dropout(sublayer(self.norm(x)))


class NaiveSublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(NaiveSublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, raw_x=None, adj=None, update_layer=None):
        transformed_x = x

        if update_layer is not None:
            hidden = transformed_x
            input = self.dropout(sublayer(self.norm(x)))
            mb, seq_len, hidden_size = hidden.size()
            hidden = hidden.view(mb * seq_len, hidden_size)
            input = input.view(mb * seq_len, hidden_size)
            new_hidden = update_layer(input, hidden)
            new_hidden = new_hidden.view(mb, seq_len, hidden_size)
            return new_hidden

        return transformed_x + self.dropout(sublayer(self.norm(x)))


class GraphNaiveSublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(GraphNaiveSublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, raw_x=None, adj=None, update_layer=None):
        transformed_x = torch.bmm(adj, x)

        if update_layer is not None:
            input = self.dropout(sublayer(self.norm(x)))
            hidden = transformed_x
            mb, seq_len, hidden_size = input.size()
            input = input.view(mb * seq_len, hidden_size)
            hidden = hidden.view(mb * seq_len, hidden_size)
            new_hidden = update_layer(input, hidden)
            new_hidden = new_hidden.view(mb, seq_len, hidden_size)
            return new_hidden

        return transformed_x + self.dropout(sublayer(self.norm(x)))


class GraphRawSublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(GraphRawSublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, raw_x=None, adj=None, update_layer=None):
        transformed_x = torch.bmm(adj, raw_x)
        # hidden: transformed_x
        # input：self.dropout(sublayer(self.norm(x)))
        # use gate for deep gcn/transformer
        # Residual or Gate? Towards Deeper Graph Neural
        # Networks for Inductive Graph Representation
        # Learning
        if update_layer is not None:
            input = self.dropout(sublayer(self.norm(x)))
            hidden = transformed_x
            mb, seq_len, hidden_size = input.size()
            input = input.view(mb * seq_len, hidden_size)
            hidden = hidden.view(mb * seq_len, hidden_size)
            new_hidden = update_layer(input, hidden)
            new_hidden = new_hidden.view(mb, seq_len, hidden_size)
            return new_hidden
        return transformed_x + self.dropout(sublayer(self.norm(x)))

