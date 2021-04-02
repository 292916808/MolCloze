# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:52
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class NNBilinearDecoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super(NNBilinearDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = F.sigmoid
        self.relation = Parameter(torch.FloatTensor(input_dim, input_dim))
        ffn = [
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        ]
        self.ffn = nn.Sequential(*ffn)
        self.reset_parameter()

    def reset_parameter(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            elif param.dim() == 0:
                nn.init.constant_(param, 1.)
            else:
                nn.init.xavier_normal_(param)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: (batch_size, seq_len, hidden_size)
        :return: (seq_len * seq_len, )
        """
        inputs_row = inputs
        inputs_col = inputs
        inputs_row = self.ffn(self.dropout(inputs_row))
        inputs_col = self.ffn(self.dropout(inputs_col)).transpose(0, 1)
        intermediate_product = torch.bmm(inputs_row, self.relation)
        rec = torch.bmm(intermediate_product, inputs_col)
        outputs = self.act(rec)
        outputs = outputs.view(-1)
        return outputs

