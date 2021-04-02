# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:59
import socket


def get_pubchem_zinc_path():
    root = '/home/wangyh' if socket.gethostname().startswith('serverx') else '/home/workspace/yongfenghuang'
    pubchem_zinc_path = '{root}/data/pubchem_zinc/'.format(root=root)
    return pubchem_zinc_path


def get_preprocessed_path():
    root = '/home/wangyh' if socket.gethostname().startswith('serverx') else '/home/workspace/yongfenghuang'
    return root + '/data/preprocessed'


def get_available_data_types():
    return ['small', '3m', '6m', '12m', '24m', 'long', '1m', '2m', 'baai']
