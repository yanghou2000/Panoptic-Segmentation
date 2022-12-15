# This file contains GlobalSAModule, SAModule

import os
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius

class SAModule(torch.nn.Module):
    '''Set Abstraction Module
    '''
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio  #ratio of points to be sampled
        self.r = r  #radius of circle in which points are grouped
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        # print(type(x), type(pos), x.size())
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

