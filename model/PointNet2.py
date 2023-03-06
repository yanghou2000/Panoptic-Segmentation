import os
import time
import yaml

import numpy as np
import torch
import torch.nn.functional as F

from Modules import GlobalSAModule, SAModule
from torch_scatter import scatter
from torch_geometric.nn import MLP, knn_interpolate
from torch_geometric.data import Data

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
        # TODO: create idx of size of data.pos and pass into sa0_out, think about batch idx 
        init_idx = torch.arange(data.batch.size(0)) 
        sa0_out = (None, data.pos, data.batch, init_idx)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        # sa3_out = self.sa3_module(*sa2_out)
        sa3_out = self.sa3_module(*sa2_out) # only use downsampled points from the previous two sa modules
        seed_idx = sa2_out[3]
        # print('seed_idx: ', seed_idx)

        # Semantic branch
        sem_fp3_out = self.sem_fp3_module(*sa3_out, *sa2_out)
        sem_fp2_out = self.sem_fp2_module(*sem_fp3_out, *sa1_out)
        sem_x, _, _, _ = self.sem_fp1_module(*sem_fp2_out, *sa0_out)

        sem_out = self.mlp(sem_x).log_softmax(dim=-1) # output probabilities for each class [N_points X N_class]

        # Instane branch
        inst_fp3_out = self.inst_fp3_module(*sa3_out, *sa2_out)
        inst_fp2_out = self.inst_fp2_module(*inst_fp3_out, *sa1_out)
        inst_out, _, _, _ = self.inst_fp1_module(*inst_fp2_out, *sa0_out) # output features [N_points X N_features]

        return sem_out, inst_out, seed_idx

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, seed_idx, x_skip, pos_skip, batch_skip, seed_idx_skip):
        # Yang: add nummy seed_idx and seed_idx_skip to keep overall code structure unchanged
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip, None


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_classes=20).to(device)
    model.eval()

    size1 = 3 * 4 # B x Np
    points = torch.randint(low=0, high=10, size=(size1, 3))
    map_sem_label = torch.randint(low=0, high=20, size=(size1,))
    inst_label = torch.randint(low=0, high=20, size=(size1,))
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int64)
    data = Data(pos=points, y=map_sem_label, z=inst_label)
    data.batch = batch

    data.to(device)

    sem_out, inst_out, seed_idx = model(data)
    print(seed_idx)