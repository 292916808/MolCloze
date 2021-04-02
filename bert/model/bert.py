# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:34
import socket
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp

from .embedding import MolBertEmbedding
from .transformer import TransformerBlock

hostname = socket.gethostname()


def normalize_adj(adj: sp.csr_matrix) -> sp.coo_matrix:
    adj = sp.coo_matrix(adj)
    # contain self-loop
    adj_ = adj + sp.eye(adj.shape[0])
    # eliminate self-loop
    # adj_ = adj
    rowsum = np.array(adj_.sum(0))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_norm


class MolBert(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1,
                 residual_type='naive', without_ffn=False):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super(MolBert, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.residual_type = residual_type
        self.without_ffn = without_ffn

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = 4 * hidden

        # embedding for Bert
        self.embedding = MolBertEmbedding(
            vocab_size=vocab_size, embed_size=hidden
        )

        # # add params for graph residual learning
        # self.residual_weights = nn.ModuleList(
        #     [Parameter(torch.FloatTensor(hidden, hidden))
        #      for _ in range(n_layers)]
        # )
        # for i in range(n_layers):
        #     stddev = 1. / math.sqrt(hidden)
        #     self.residual_weights[i].data.uniform_(-stddev, stddev)

        # add params for gated learning implemented by LSTM
        # self.update_layer = nn.GRUCell(hidden, hidden)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout=dropout,
                              residual_type=residual_type, without_ffn=without_ffn)
             for _ in range(n_layers)]
        )

    def forward(self, x, segment_info, adj=None):
        # attention masking for padded token
        # (batch_size, seq_len, seq_len)
        temp_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1)
        temp_mask_t = temp_mask.transpose(1, 2)
        mask = temp_mask * temp_mask_t
        mask = mask.unsqueeze(1)
        # print(mask)
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        # add adjacency matrix as a constraint
        # adj: (batch_size, 1, seq_len, seq_len)
        if adj is not None:
            if hostname.startswith('serverx51'):
                adj = adj.unsqueeze(1)
                mask = mask.float()
            else:
                adj = adj.unsqueeze(1).byte()
            mask *= adj
            # adj: (batch_size, seq_len, seq_len)
            adj = adj.squeeze(1).float().detach().cpu().numpy()
            adj = [torch.FloatTensor(normalize_adj(sub_adj).todense()) for sub_adj in adj]
            adj = torch.stack(adj, dim=0)
            if torch.cuda.is_available():
                adj = adj.cuda()

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)
        # print(x.size())
        outputs = []

        # running over multiple transformer blocks
        raw_x = x.clone()

        # modify:mask = None
        # mask = None
        for i, transformer in enumerate(self.transformer_blocks):
            x = transformer.forward(x, mask, raw_x, adj)
            outputs.append(x)
        # print(x.size())
        return x, outputs

    def get_input_embedding(self):
        """
        Get model's input embeddings
        """
        return self.embedding.token

    def set_input_embeddings(self, value):
        """
        Set model's input embeddings
        """
        self.embedding.token = value

    def get_output_embeddings(self):
        """ Get model's output embeddings
            Return None if the model doesn't have output embeddings
        """
        return None

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if hasattr(output_embeddings, 'bias') and output_embeddings.bias is not None:
            output_embeddings.bias.data = torch.nn.functional.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
                'constant',
                0
            )
        if hasattr(output_embeddings, 'out_features') and hasattr(input_embeddings, 'num_embeddings'):
            output_embeddings.out_features = input_embeddings.num_embeddings

    @classmethod
    def from_pretrained(cls, pretrained_model_path, **model_kwargs):
        output_loading_info = model_kwargs.get('output_loading_info', None)
        # Instantiate model.
        model = cls(**model_kwargs)

        state_dict = torch.load(pretrained_model_path, map_location='cpu')

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key == 'lm_head.decoder.weight':
                new_key = 'm_head.weight'
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        mdoel_to_load = model
        load(mdoel_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))

        if hasattr(model, 'tie_weights'):
            model.tie_weights()  # make sure word embedding weights are still tied

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model
