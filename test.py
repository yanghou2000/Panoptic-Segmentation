import torch_cluster
import os
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius

input = torch.randn(3, 5, requires_grad=True)
target = torch.tensor([1, 0, 4])
output = F.nll_loss(F.log_softmax(input, dim=1), target)
output.backward()