# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:37
import torch.nn as nn
from .token import TokenEmbedding
from .segment import SegmentEmbedding


class MolBertEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super(MolBertEmbedding, self).__init__()
        self.token = TokenEmbedding(vocab_size, embed_size)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):       
        x = self.token(sequence) + self.segment(segment_label)
        
        return self.dropout(x)
