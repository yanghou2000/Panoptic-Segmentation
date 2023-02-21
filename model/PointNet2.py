import os
import time
import yaml

import numpy as np
import torch
import torch.nn.functional as F
from Modules import GlobalSAModule, SAModule
from torch_scatter import scatter
from torch_geometric.nn import MLP, knn_interpolate

class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.02, 0.2, MLP([3, 64, 64, 128])) # ratio, r, nn
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.sem_fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.sem_fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.sem_fp1_module = FPModule(3, MLP([128, 128, 128, 128]))
        # self.sem_fp1_module = FPModule(3, MLP([128+3, 128, 128, 128]))

        self.inst_fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.inst_fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        # Yang: change the last layer of inst_fp1_modul from 128 to 64 to reduce size
        self.inst_fp1_module = FPModule(3, MLP([128, 128, 128, 64]))

        self.mlp = MLP([128, 128, 128, num_classes], dropout=0.5, norm=None)


    def forward(self, data):
        # Shared encoder
        sa0_out = (None, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        # Semantic branch
        sem_fp3_out = self.sem_fp3_module(*sa3_out, *sa2_out)
        sem_fp2_out = self.sem_fp2_module(*sem_fp3_out, *sa1_out)
        sem_x, _, _ = self.sem_fp1_module(*sem_fp2_out, *sa0_out)

        sem_out = self.mlp(sem_x).log_softmax(dim=-1) # output probabilities for each class [N_points X N_class]

        # Instane branch
        inst_fp3_out = self.inst_fp3_module(*sa3_out, *sa2_out)
        inst_fp2_out = self.inst_fp2_module(*inst_fp3_out, *sa1_out)
        inst_out, _, _ = self.inst_fp1_module(*inst_fp2_out, *sa0_out) # output features [N_points X N_features]

        return sem_out, inst_out

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip