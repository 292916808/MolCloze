# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:31

import random


class DeepWalker(object):
    def __init__(self, G, num_workers=4):
        self.G = G
        self.num_workers = num_workers

    def walk(self, walk_length, start_node):
        G = self.G

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            neighbors = list(G.neighbors(cur))
            if len(neighbors) > 0:
                walk.append(random.choice(neighbors))
            else:
                break
        return walk

