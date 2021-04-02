# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:34

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .bert import MolBert



class Config(object):
    def __init__(self,
                 vocab_size_or_config_json_file,
                 act_fn="gelu",
                 hidden_size=768,
                 embedding_size=768,
                 num_hidden_layers=12,
                 num_heads=12,
                 dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 loss_weight=1.0,
                 initializer_range=0.02
                 ):
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.act_fn = act_fn
            self.hidden_size = hidden_size
            self.embedding_size = embedding_size
            self.num_hidden_layers = num_hidden_layers
            self.num_heads = num_heads
            self.dropout_prob = dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.loss_weight = loss_weight
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        config = Config(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.shape = torch.Size((hidden_size,))
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def fused_layer_norm(self, x):
        return FusedLayerNormAffineFunction.apply(
            x, self.weight, self.bias, self.shape, self.eps)

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


LayerNorm = BertLayerNorm


def Linear(i_dim, o_dim, bias=True):
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


def bias_tanh(bias, y):
    x = bias + y
    return torch.tanh(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "bias_gelu": bias_gelu, "bias_tanh": bias_tanh, "relu": torch.nn.functional.relu, "swish": swish}


class LinearActivation(nn.Module):
    r"""Fused Linear and activation Module.
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, act='gelu', bias=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_fn = nn.Identity()                                                         #
        self.biased_act_fn = None                                                           #
        self.bias = None                                                                    #
        if isinstance(act, str) or (sys.version_info[0] == 2 and isinstance(act, unicode)): # For TorchScript
            if bias and not 'bias' in act:                                                  # compatibility
                act = 'bias_' + act                                                         #
                self.biased_act_fn = ACT2FN[act]                                            #

            else:
                self.act_fn = ACT2FN[act]
        else:
            self.act_fn = act
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if not self.bias is None:
            return self.biased_act_fn(self.bias, F.linear(input, self.weight, None))
        else:
            return self.act_fn(F.linear(input, self.weight, self.bias))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class MaskedLanguageModel(nn.Module):

    def __init__(self, hidden, vocab_size):
        super(MaskedLanguageModel, self).__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class NextSentencePrediction(nn.Module):

    def __init__(self, hidden):
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))

class MolBertLM(nn.Module):

    def __init__(self, bert, vocab_size):
        super(MolBertLM, self).__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size=vocab_size)
        # self.gen_preds = ElectraGeneratorPredictionHeads(self.bert.hidden)

    def forward(self, x, segment_label, adj=None):
        x, outputs = self.bert(x, segment_label, adj)
        
        return self.mask_lm(x), outputs, x

class ElectraDiscriminatorPredictionHeads(nn.Module):
    def __init__(self, bert):
        super(ElectraDiscriminatorPredictionHeads, self).__init__()
        self.bert = bert
        self.dense = LinearActivation(self.bert.hidden, self.bert.hidden)
        self.dense_prediction = Linear(self.bert.hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dense_prediction(hidden_states)
        return self.sigmoid(hidden_states)

class MolBertDis(nn.Module):
    def __init__(self, bert):
        super(MolBertDis, self).__init__()
        self.bert = bert
        self.predictions = ElectraDiscriminatorPredictionHeads(self.bert)

    def forward(self, x, segment_label, adj=None):
        x, outputs = self.bert(x, segment_label, adj)
        prediction_scores = self.predictions(x).squeeze(-1)
        return prediction_scores, x

# class ElectraModel(nn.Module):
#     def __init__(self)