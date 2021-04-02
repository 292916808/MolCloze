# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# datetime: 2020/4/20 15:31
import random

class DFSWalker(object):
    def __init__(self, G, num_workers=4):
        self.G = G
        self.num_workers = num_workers

    def walk(self, walk_length, start_node):

        def try_walk(walk, walk_length):
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = list(G.neighbors(cur))
                # ensure that the nodes to be added is not visited before.
                neighbors = [node for node in neighbors if node not in walk]
                if len(neighbors) > 0:
                    node = random.choice(neighbors)
                    walk.append(node)
                else:
                    break
            return walk

        G = self.G
        # sever as queue
        walk = [start_node]
        walk = try_walk(walk, walk_length)
        if len(walk) > walk_length:
            walk = walk[:walk_length]
        return walk


