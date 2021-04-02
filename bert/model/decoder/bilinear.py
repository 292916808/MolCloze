# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:51
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BilinearDecoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super(BilinearDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = torch.sigmoid
        self.relation = Parameter(torch.FloatTensor(input_dim, input_dim))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.relation.data)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: (batch_size, seq_len, hidden_size)
        :return: (seq_len * seq_len, )
        """
        batch_size = inputs.size(0)
        inputs_row = inputs
        inputs_col = inputs.transpose(1, 2)
        inputs_row = self.dropout(inputs_row)
        inputs_col = self.dropout(inputs_col)
        relation = self.relation.unsqueeze(0).repeat(batch_size, 1, 1)
        if torch.cuda.is_available():
            relation = relation.cuda()
        intermediate_product = torch.bmm(inputs_row, relation)
        rec = torch.bmm(intermediate_product, inputs_col)
        outputs = self.act(rec)
        outputs = outputs.view(-1)
        return outputs
