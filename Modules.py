# This file contains GlobalSAModule, SAModule

from numbers import Number
from typing import Tuple

import os
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius
from torch_geometric.nn.conv import MessagePassing

from torch import Tensor, LongTensor
from torch.nn import Linear
# from torch_geometric.nn.pool.decimation import decimation_indices
# from testing import decimation_indices
from torch_geometric.utils import softmax
from torch_geometric.nn.pool import knn_graph
# from torch_cluster import knn_graph
# from testing import knn_graph
from torch_geometric.nn import knn_interpolate
from torch_geometric.utils import scatter
lrelu02_kwargs = {'negative_slope': 0.2}
bn099_kwargs = {'momentum': 0.01, 'eps': 1e-6}
from utils.utils import set_random_seeds

set_random_seeds(42)

# PointNet++
class SAModule(torch.nn.Module):
    '''Set Abstraction Module
    '''
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio  #ratio of points to be sampled
        self.r = r  #radius of circle in which points are grouped
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch, seed_idx):
        # Yang: add idx of downsampled points to the output
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        if seed_idx is None:
            seed_idx = None
        else:
            seed_idx = seed_idx[idx]
        pos, batch = pos[idx], batch[idx] 
        return x, pos, batch, seed_idx

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch, seed_idx):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch, None

# RandlaNet
class SharedMLP(MLP):
    # Default activation and batch norm parameters used by RandLA-Net:
    """SharedMLP following RandLA-Net paper."""
    def __init__(self, *args, **kwargs):
        # BN + Act always active even at last layer.
        kwargs['plain_last'] = False
        # LeakyRelu with 0.2 slope by default.
        kwargs['act'] = kwargs.get('act', 'LeakyReLU')
        kwargs['act_kwargs'] = kwargs.get('act_kwargs', lrelu02_kwargs)
        # BatchNorm with 1 - 0.99 = 0.01 momentum
        # and 1e-6 eps by defaut (tensorflow momentum != pytorch momentum)
        kwargs['norm_kwargs'] = kwargs.get('norm_kwargs', bn099_kwargs)
        super().__init__(*args, **kwargs)


class LocalFeatureAggregation(MessagePassing):
    """Positional encoding of points in a neighborhood."""
    def __init__(self, channels):
        super().__init__(aggr='add')
        self.mlp_encoder = SharedMLP([10, channels // 2])
        self.mlp_attention = SharedMLP([channels, channels], bias=False,
                                       act=None, norm=None)
        self.mlp_post_attention = SharedMLP([channels, channels])

    def forward(self, edge_index, x, pos):
        out = self.propagate(edge_index, x=x, pos=pos)  # N, d_out
        out = self.mlp_post_attention(out)  # N, d_out
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                index: Tensor) -> Tensor:
        """Local Spatial Encoding (locSE) and attentive pooling of features.
        Args:
            x_j (Tensor): neighboors features (K,d)
            pos_i (Tensor): centroid position (repeated) (K,3)
            pos_j (Tensor): neighboors positions (K,3)
            index (Tensor): index of centroid positions
                (e.g. [0,...,0,1,...,1,...,N,...,N])
        returns:
            (Tensor): locSE weighted by feature attention scores.
        """
        # Encode local neighboorhod structural information
        pos_diff = pos_j - pos_i
        distance = torch.sqrt((pos_diff * pos_diff).sum(1, keepdim=True))
        relative_infos = torch.cat([pos_i, pos_j, pos_diff, distance],
                                   dim=1)  # N * K, d
        local_spatial_encoding = self.mlp_encoder(relative_infos)  # N * K, d
        local_features = torch.cat([x_j, local_spatial_encoding],
                                   dim=1)  # N * K, 2d

        # Attention will weight the different features of x
        # along the neighborhood dimension.
        att_features = self.mlp_attention(local_features)  # N * K, d_out
        att_scores = softmax(att_features, index=index)  # N * K, d_out

        return att_scores * local_features  # N * K, d_out


class DilatedResidualBlock(torch.nn.Module):
    def __init__(
        self,
        num_neighbors,
        d_in: int,
        d_out: int,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.d_in = d_in
        self.d_out = d_out

        # MLP on input
        self.mlp1 = SharedMLP([d_in, d_out // 8])
        # MLP on input, and the result is summed with the output of mlp2
        self.shortcut = SharedMLP([d_in, d_out], act=None)
        # MLP on output
        self.mlp2 = SharedMLP([d_out // 2, d_out], act=None)

        self.lfa1 = LocalFeatureAggregation(d_out // 4)
        self.lfa2 = LocalFeatureAggregation(d_out // 2)

        self.lrelu = torch.nn.LeakyReLU(**lrelu02_kwargs)

    def forward(self, x, pos, batch):
        edge_index = knn_graph(pos, self.num_neighbors, batch=batch, loop=True)

        shortcut_of_x = self.shortcut(x)  # N, d_out
        x = self.mlp1(x)  # N, d_out//8
        x = self.lfa1(edge_index, x, pos)  # N, d_out//2
        x = self.lfa2(edge_index, x, pos)  # N, d_out//2
        x = self.mlp2(x)  # N, d_out
        x = self.lrelu(x + shortcut_of_x)  # N, d_out

        # print('pos: ', pos)
        return x, pos, batch

def decimation_indices(ptr: LongTensor,
                       decimation_factor: Number) -> Tuple[Tensor, LongTensor]:
    """Get indices which downsample each point cloud by a decimation factor.

    Decimation happens separately for each cloud to prevent emptying smaller
    point clouds. Empty clouds are prevented: clouds will have a least
    one node after decimation.

    Args:
        ptr (LongTensor): indices of samples in the batch.
        decimation_factor (Number): value to divide number of nodes with.
            Should be higher than 1 for downsampling.

    :rtype: (:class:`Tensor`, :class:`LongTensor`): indices for downsampling
        and resulting updated ptr.

    """
    if decimation_factor < 1:
        raise ValueError(
            "Argument `decimation_factor` should be higher than (or equal to) "
            f"1 for downsampling. (Current value: {decimation_factor})")

    batch_size = ptr.size(0) - 1
    bincount = ptr[1:] - ptr[:-1]
    decimated_bincount = torch.div(bincount, decimation_factor,
                                   rounding_mode="floor")
    # Decimation should not empty clouds completely.
    decimated_bincount = torch.max(torch.ones_like(decimated_bincount),
                                   decimated_bincount)
    idx_decim = torch.cat(
        [(ptr[i] + torch.randperm(bincount[i],
                                  device=ptr.device)[:decimated_bincount[i]])
         for i in range(batch_size)],
        dim=0,
    )
    # Get updated ptr (e.g. for future decimations)
    ptr_decim = ptr.clone()
    for i in range(batch_size):
        ptr_decim[i + 1] = ptr_decim[i] + decimated_bincount[i]

    return idx_decim, ptr_decim



def decimate(tensors, ptr: Tensor, decimation_factor: int):
    """Decimates each element of the given tuple of tensors."""
    idx_decim, ptr_decim = decimation_indices(ptr, decimation_factor)

    tensors_decim = tuple(tensor[idx_decim] for tensor in tensors)
    return tensors_decim, ptr_decim, idx_decim



if __name__=='__main__':
    tensors = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    ptr = torch.tensor([0, 1, 2])
    tensors_decim, ptr_decim = decimate(tensors, ptr, 2)
    print(tensors_decim, ptr_decim)