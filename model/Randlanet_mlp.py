# This file is copied from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/randlanet_segmentation.py
# My contribution is to add instance branches such that RandlaNet can perform not only semantic but laso intance segmentation
"""
An implementation of RandLA-Net based on the paper:
RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds
Reference: https://arxiv.org/abs/1911.11236
"""
import os.path as osp

import torch
import torch.nn.functional as F
from Modules import DilatedResidualBlock, SharedMLP, decimate
from torch.nn import Linear

from torch_geometric.nn import knn_interpolate, MLP
from torch_geometric.utils import scatter

from torch_geometric.data import Dataset, Data
from model.ASIS import ASIS

class RandlaNet_mlp(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        decimation: int = 4,
        num_neighbors: int = 16,
        return_logits: bool = False,
        dim_inst_out: int = 32,
        use_asis: bool = False,
        asis: dict = {'feature_dim': 32, 'inst_emb_dim': 5, 'num_class': 20, 'k':30, 'distance_threshold': 0.5, 'norm_degree': 2}
    ):
        super().__init__()

        self.decimation = decimation
        # An option to return logits instead of log probabilities:
        self.return_logits = return_logits
        self.use_asis = use_asis
        keys = ['feature_dim', 'inst_emb_dim', 'num_class', 'k', 'distance_threshold', 'norm_degree']
        [feature_dim, inst_emb_dim, num_class, k, distance_threshold, norm_degree] = [asis[key] for key in keys]
        self.asis = ASIS(feature_dim=feature_dim, inst_emb_dim=inst_emb_dim, num_class=num_class, k=k, distance_threshold=distance_threshold, norm_degree=norm_degree) # distance_threshold = delta_var
        # Authors use 8, which is a bottleneck
        # for the final MLP, and also when num_classes>8
        # or num_features>8.
        # d_bottleneck = max(32, num_classes, num_features)
        d_bottleneck = 8

        self.fc0 = Linear(num_features, d_bottleneck)
        self.block1 = DilatedResidualBlock(num_neighbors, d_bottleneck, 32) # num_neighbors, d_in, d_out
        self.block2 = DilatedResidualBlock(num_neighbors, 32, 128)
        self.block3 = DilatedResidualBlock(num_neighbors, 128, 256)
        self.block4 = DilatedResidualBlock(num_neighbors, 256, 512)
        self.mlp_summit = SharedMLP([512, 512])
        
        # semnatic branch
        self.sem_fp4 = FPModule(1, SharedMLP([512 + 256, 256]))
        self.sem_fp3 = FPModule(1, SharedMLP([256 + 128, 128]))
        self.sem_fp2 = FPModule(1, SharedMLP([128 + 32, 32]))
        self.sem_fp1 = FPModule(1, SharedMLP([32 + 32, d_bottleneck]))

        # instance branch
        self.inst_fp4 = FPModule(1, SharedMLP([512 + 256, 256]))
        self.inst_fp3 = FPModule(1, SharedMLP([256 + 128, 128]))
        self.inst_fp2 = FPModule(1, SharedMLP([128 + 32, 32]))
        self.inst_fp1 = FPModule(1, SharedMLP([32 + 32, d_bottleneck]))

        self.mlp_classif = SharedMLP([d_bottleneck, 64, 32],
                                     dropout=[0.0, 0.5], norm='batch_norm')
        # self.mlp_classif_0 = SharedMLP([d_bottleneck, 64], dropout=0.0)
        # self.mlp_classif_1 = SharedMLP([64, 32], dropout-0.5, plain_last=True)
        self.fc_classif = Linear(32, num_classes)

        # dowansample the instance features into 5 dimensions
        # self.fc_inst = SharedMLP([d_bottleneck, dim_inst_out])
        self.fc_inst = Linear(d_bottleneck, dim_inst_out)

        # Freeze the first convolutional layer
        freeze_list = [self.sem_fp1, self.sem_fp2, self.sem_fp3, self.sem_fp4, self.fc_classif, self.mlp_classif]
        for layer in freeze_list:
           for param in layer.parameters():
                param.requires_grad = False

    def forward(self, data):
        data.x = data.x if data.x is not None else data.pos

        initial_seed_idx = torch.arange(data.x.shape[0])

        b1_out = self.block1(self.fc0(data.x), data.pos, data.batch)
        b1_out_decimated, ptr1, idx_1 = decimate(b1_out, data.ptr, self.decimation) # downsampling
        # print('b1_out: ', b1_out, len(b1_out), 'b1_out_decimated', b1_out_decimated, len(b1_out_decimated), ptr1, ptr1.shape)
        # print(ptr1, ptr1.shape)

        b2_out = self.block2(*b1_out_decimated)
        b2_out_decimated, ptr2, idx_2 = decimate(b2_out, ptr1, self.decimation)
        # print(b2_out, b2_out.shape, b2_out_decimated, b2_out_decimated.shape, ptr2, ptr2.shape)
        # print(ptr2, ptr2.shape)

        b3_out = self.block3(*b2_out_decimated)
        b3_out_decimated, ptr3, idx_3 = decimate(b3_out, ptr2, self.decimation)
        # print(b3_out, b3_out.shape, b3_out_decimated, b3_out_decimated.shape, ptr3, ptr3.shape)
        # print(ptr3, ptr3.shape)

        b4_out = self.block4(*b3_out_decimated)
        b4_out_decimated, ptr4, idx_4 = decimate(b4_out, ptr3, self.decimation)
        # print(b4_out, b4_out.shape, b4_out_decimated, b4_out_decimated.shape, ptr4, ptr4.shape)
        # print(ptr4, ptr4.shape)

        seed_idx = initial_seed_idx[idx_1][idx_2][idx_3][idx_4]

        mlp_out = (
            self.mlp_summit(b4_out_decimated[0]),
            b4_out_decimated[1],
            b4_out_decimated[2],
        )

        # Semantic branch
        sem_fp4_out = self.sem_fp4(*mlp_out, *b3_out_decimated)
        sem_fp3_out = self.sem_fp3(*sem_fp4_out, *b2_out_decimated)
        sem_fp2_out = self.sem_fp2(*sem_fp3_out, *b1_out_decimated)
        sem_fp1_out = self.sem_fp1(*sem_fp2_out, *b1_out)

        sem_x = self.mlp_classif(sem_fp1_out[0])
        
        # Instance branch
        inst_fp4_out = self.inst_fp4(*mlp_out, *b3_out_decimated)
        inst_fp3_out = self.inst_fp3(*inst_fp4_out, *b2_out_decimated)
        inst_fp2_out = self.inst_fp2(*inst_fp3_out, *b1_out_decimated)
        inst_fp1_out = self.inst_fp1(*inst_fp2_out, *b1_out)

        if self.use_asis == True:
            sem_logits, inst_out = self.asis(sem_x, inst_fp1_out[0], data.batch)
            sem_out = sem_logits.log_softmax(dim=-1)
        else:
            sem_logits = self.fc_classif(sem_x)
            sem_out = sem_logits.log_softmax(dim=-1)
            # inst_out = inst_fp1_out[0]
            inst_out = self.fc_inst(inst_fp1_out[0])

        return sem_out, inst_out, seed_idx


class FPModule(torch.nn.Module):
    """Upsampling with a skip connection."""
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip