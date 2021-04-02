# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:50
import torch
import torch.nn as nn


class InnerProductDecoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = torch.sigmoid
        self.criterion = nn.MSELoss()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: (batch_size, seq_len, hidden_size)
        :return: (seq_len * seq_len, )
        """
        # temp_mask = (targets > 0).unsqueeze(1).repeat(1, targets.size(1), 1)
        # temp_mask_t = temp_mask.transpose(1, 2)
        # mask = temp_mask * temp_mask_t
        # lengths = [torch.sum(sub_mask[0]).item() for sub_mask in mask]
        inputs_row = inputs
        inputs_col = inputs.transpose(1, 2)
        inputs_row = self.dropout(inputs_row)
        inputs_col = self.dropout(inputs_col)
        rec = torch.bmm(inputs_row, inputs_col)
        outputs = self.act(rec)
        outputs = outputs.view(-1)
        return outputs
        # new_rec = []
        # new_targets = []
        # for length, single_rec, single_adj in zip(lengths, rec, adj):
        #     new_rec.extend(single_rec[:length, :length].reshape(length * length, ))
        #     new_targets.extend(single_adj[:length, :length].reshape(length * length, ))
        # new_rec = torch.tensor(new_rec)
        # new_targets = torch.tensor(new_targets)
        # new_rec = self.act(new_rec)
        # dec_loss = self.criterion(new_rec, new_targets)
        # return dec_loss