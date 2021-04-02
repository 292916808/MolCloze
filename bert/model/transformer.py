# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:34
import torch
import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward
from .utils import NaiveSublayerConnection, RawSublayerConnection, GraphNaiveSublayerConnection, GraphRawSublayerConnection


class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout,
                residual_type, without_ffn):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hdiden, usually 4 * hidden
        :param dropout: dropout rate
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        if residual_type == 'naive':
            self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        elif residual_type == 'raw':
            self.input_sublayer = RawSublayerConnection(size=hidden, dropout=dropout)
        elif residual_type == 'graph_naive':
            self.input_sublayer = GraphNaiveSublayerConnection(size=hidden, dropout=dropout)
        elif residual_type == 'graph_raw':
            self.input_sublayer = GraphRawSublayerConnection(size=hidden, dropout=dropout)
        else:
            self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout) if not without_ffn else None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, raw_x=None, adj=None):
        x = self.input_sublayer(x, lambda x_: self.attention.forward(x_, x_, x_, mask=mask),
                                raw_x, adj)
        if self.output_sublayer is not None:
            x = self.output_sublayer(x, self.feed_forward,
                                     raw_x, adj)
        else:
            x = self.dropout(torch.relu(x))
        return self.dropout(x)


