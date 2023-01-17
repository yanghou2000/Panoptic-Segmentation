# # for testing null_loss functionality
# import torch_cluster
# import os
# import torch
# import torch.nn.functional as F

# import torch_geometric.transforms as T
# from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius

# input = torch.randn(3, 5, requires_grad=True)
# target = torch.tensor([1, 0, 4])
# output = F.nll_loss(F.log_softmax(input, dim=1), target)
# output.backward()

# # for testing category,data.ptr functions
# import os.path as osp

# import torch
# import torch.nn.functional as F
# from Modules import GlobalSAModule, SAModule
# from torch_scatter import scatter
# from torchmetrics.functional import jaccard_index

# import torch_geometric.transforms as T
# from torch_geometric.datasets import ShapeNet
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import MLP, knn_interpolate

# category = 'Airplane'  # Pass in `None` to train on all categories.
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ShapeNet')
# transform = T.Compose([
#     T.RandomJitter(0.01),
#     T.RandomRotate(15, axis=0),
#     T.RandomRotate(15, axis=1),
#     T.RandomRotate(15, axis=2)
# ])
# pre_transform = T.NormalizeScale()
# train_dataset = ShapeNet(path, category, split='trainval', transform=transform,
#                          pre_transform=pre_transform)
# test_dataset = ShapeNet(path, category, split='test',
#                         pre_transform=pre_transform)
# train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True,
#                           num_workers=6)
# test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False,
#                          num_workers=6)


# class FPModule(torch.nn.Module):
#     def __init__(self, k, nn):
#         super().__init__()
#         self.k = k
#         self.nn = nn

#     def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
#         x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
#         if x_skip is not None:
#             x = torch.cat([x, x_skip], dim=1)
#         x = self.nn(x)
#         return x, pos_skip, batch_skip


# class Net(torch.nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         # Input channels account for both `pos` and node features.
#         self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128]))
#         self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
#         self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

#         self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
#         self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
#         self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, 128]))

#         self.mlp = MLP([128, 128, 128, num_classes], dropout=0.5, norm=None)

#         self.lin1 = torch.nn.Linear(128, 128)
#         self.lin2 = torch.nn.Linear(128, 128)
#         self.lin3 = torch.nn.Linear(128, num_classes)

#     def forward(self, data):
#         sa0_out = (data.x, data.pos, data.batch)
#         sa1_out = self.sa1_module(*sa0_out)
#         sa2_out = self.sa2_module(*sa1_out)
#         sa3_out = self.sa3_module(*sa2_out)

#         fp3_out = self.fp3_module(*sa3_out, *sa2_out)
#         fp2_out = self.fp2_module(*fp3_out, *sa1_out)
#         x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

#         return self.mlp(x).log_softmax(dim=-1)


# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# model = Net(train_dataset.num_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# def train():
#     model.train()

#     total_loss = correct_nodes = total_nodes = 0
#     for i, data in enumerate(train_loader):
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data)
#         loss = F.nll_loss(out, data.y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
#         total_nodes += data.num_nodes

#         if (i + 1) % 10 == 0:
#             print(f'[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} '
#                   f'Train Acc: {correct_nodes / total_nodes:.4f}')
#             total_loss = correct_nodes = total_nodes = 0


# @torch.no_grad()
# def test(loader):
#     model.eval()

#     ious, categories = [], []
#     y_map = torch.empty(loader.dataset.num_classes, device=device).long()
#     for data in loader:
#         data = data.to(device)
#         outs = model(data)

#         # for debugging
#         print("data.ptr =", data.ptr)
#         print("sizes =", sizes)
#         print("outs.split(sizes) =", outs.split(sizes))
#         quit()

#         sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
#         for out, y, category in zip(outs.split(sizes), data.y.split(sizes),
#                                     data.category.tolist()):
#             category = list(ShapeNet.seg_classes.keys())[category]
#             part = ShapeNet.seg_classes[category]
#             part = torch.tensor(part, device=device)

#             y_map[part] = torch.arange(part.size(0), device=device)

#             iou = jaccard_index(out[:, part].argmax(dim=-1), y_map[y],
#                                 num_classes=part.size(0), absent_score=1.0)
#             ious.append(iou)

#         categories.append(data.category)

#     iou = torch.tensor(ious, device=device)
#     category = torch.cat(categories, dim=0)

#     mean_iou = scatter(iou, category, reduce='mean')  # Per-category IoU.
#     return float(mean_iou.mean())  # Global IoU.


# for epoch in range(1, 2):
#     # train()
#     iou = test(test_loader)
#     print(f'Epoch: {epoch:02d}, Test IoU: {iou:.4f}')


# output log
# data.ptr = tensor([    0,  2591,  5065,  7839, 10381, 13113, 15837, 18379, 20981, 23496,
#         25931, 28523, 31127])
# sizes = [2591, 2474, 2774, 2542, 2732, 2724, 2542, 2602, 2515, 2435, 2592, 2604]
# data.category = tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# category Airplane
# part tensor([0, 1, 2, 3])
# y_map[part] tensor([0, 1, 2, 3])
# iou tensor(0.0512)
# out[:, part] tensor([[-3.9814, -3.9208, -3.8693, -3.9021],
#         [-3.9819, -3.9208, -3.8699, -3.9017],
#         [-3.9808, -3.9238, -3.8727, -3.9052],
#         ...,
#         [-3.9818, -3.9211, -3.8706, -3.9023],
#         [-3.9806, -3.9229, -3.8721, -3.9047],
#         [-3.9814, -3.9214, -3.8708, -3.9023]])
# out[:, part].argmax(dim=-1) tensor([2, 2, 2,  ..., 2, 2, 2])
# y_map[y] tensor([3, 0, 1,  ..., 2, 0, 0])
import os
import time
from torch.utils.tensorboard import SummaryWriter
import torch

from utils import utils
from Semantic_Dataloader import SemanticKittiGraph

data_path = '/Volumes/scratchdata/kitti/dataset/'
# DATA_path = '/home/yanghou/project/Panoptic-Segmentation/semantic-kitti.yaml' #
DATA_path = './semantic-kitti.yaml' # for running in docker

testing_sequences = ['00']

train_sequences = ['00', '01', '02', '03', '04', '05', '06']
val_sequences = ['07']
test_sequences = ['08', '09', '10']

train_dataset = SemanticKittiGraph(dataset_dir='/Volumes/scratchdata/kitti/dataset/', 
                                sequences= train_sequences, 
                                DATA_dir=DATA_path)

train_dataset.get_xentropy_class_string(0)

run = str(0)
log_path = 'draft_log'
log_path = os.path.join(log_path, run) # train/test info path
writer = SummaryWriter(log_dir=log_path, filename_suffix=time.strftime("%Y%m%d_%H%M%S"))

# for i in range(1000):
#     writer.add_scalar('train/loss', i, i)


def add_iou(writer,ious,epoch):
    for ncls in range(len(ious)):
        # TODO: write class number to class str function
        class_name = utils.get_xentropy_class_string(ncls, DATA_path)
        # writer.add_scalar(f'IoU_{class_name}',ious,epoch)

        # by defualt label 0 is ignored, so iou for label 0 should be 0
        writer.add_scalar(f'IoU_{ncls}_{class_name}', ious[ncls], epoch)

ious_1 = torch.tensor([0.0000, 0.6543, 0.0000, 0.0000, 0.0000, 0.0676, 0.0000, 0.0000,    float('nan'),
        0.8150, 0.1433, 0.7099,    float('nan'), 0.8189, 0.1303, 0.5048, 0.4571, 0.0106,
        0.3027, 0.0802])

ious_2 = torch.tensor([0.3763, 0.4762, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,    float('nan'),
        0.6402, 0.0000, 0.5277,    float('nan'), 0.8976, 0.0000, 0.7143, 0.0000, 0.7752,
        0.3821, 0.0000])

ious = [ious_1, ious_2]

for i, i_iou in enumerate(ious):

    add_iou(writer, i_iou, i)
