# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:37
import torch.nn as nn


class TokenEmbedding(nn.Embedding):

    def __init__(self, vocab_size, embed_size=512):
        super(TokenEmbedding, self).__init__(vocab_size, embed_size, padding_idx=0)

    # def embedding_tables():
    #     return nn.Embedding()

