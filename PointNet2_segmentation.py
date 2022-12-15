import os
import time
import yaml

import numpy as np
import torch
import torch.nn.functional as F
from Modules import GlobalSAModule, SAModule
from torch_scatter import scatter

import torch_geometric.transforms as T
from Semantic_Dataloader import SemanticKittiGraph
from torch_geometric.loader import DataLoader

from torch_geometric.data import Dataset, Data
from torch.utils.tensorboard import SummaryWriter

from utils import tboard, utils
from model.PointNet2 import Net

# define the gpu index to train
os.environ["CUDA_VISIBLE_DEVICES"]="1"

data_path = '/Volumes/scratchdata/kitti/dataset/'
# DATA_path = '/home/yanghou/project/Panoptic-Segmentation/semantic-kitti.yaml' #
DATA_path = './semantic-kitti.yaml' # for running in docker
save_path = './run'

testing_squences = ['00']

train_sequences = ['00', '01', '02', '03', '04', '05', '06']
val_sequences = ['07']
test_sequences = ['08', '09', '10']

train_dataset = SemanticKittiGraph(dataset_dir='/Volumes/scratchdata/kitti/dataset/', 
                                sequences= testing_squences, 
                                DATA_dir=DATA_path)

test_dataset = SemanticKittiGraph(dataset_dir='/Volumes/scratchdata/kitti/dataset/', 
                                sequences= val_sequences, 
                                DATA_dir=DATA_path)

torch.manual_seed(42)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# add loss weights beforehand
loss_w = train_dataset.map_loss_weight()

# create a SummaryWriter to write logs to a log directory
writer = SummaryWriter('logs')

# device = torch.device('cpu') # only for debugging
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.get_n_classes()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
nloss = torch.nn.NLLLoss(weight=loss_w).to(device)

# TODO: ignore class label 0
def train():
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # negative log likelihood loss. Input should be log-softmax
        loss = nloss(out, data.y.cuda())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes
        print('Loss: ', loss)

        if (i + 1) % 10 == 0:
            print(f'[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} '
                  f'Train Acc: {correct_nodes / total_nodes:.4f}')
            total_loss = correct_nodes = total_nodes = 0
        # Log a scalar value (scalar summary)
    loss = total_loss / i
    return loss
        


# output iou and loss
@torch.no_grad()
def test(loader, model):

    model.eval()

    ious, mious = [], []
    y_map = torch.empty(loader.dataset.get_n_classes(), device=device).long()
    data_category = list(range(20))
    for data in loader:
        data = data.to(device)
        outs = model(data)

        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for out, y in zip(outs.split(sizes), data.y.split(sizes)):

            print('out.size: ', out.size(), 'y.size: ', y.size())
            iou = utils.calc_iou(out.argmax(dim=-1), y, num_classes=20)

            miou = utils.calc_miou(iou, 20)

            print('iou: ', iou, 'miou: ', miou)
            ious.append(iou)
            mious.append(miou)

        ious_avr = torch.sum(ious) / len(ious)
        miou_avr = torch.sum(mious) / len(miou)

        return ious_avr, miou_avr


# for epoch in range(1, 31):
#     loss = train()
#     ious, miou = test(test_loader)
#     writer.add_scalar('train/loss', loss, epoch)
#     writer.add_scalar('train/iou', miou, epoch)
#     tboard.add_iou(writer, ious, epoch)
#     state = {'net':model.state_dict(), 'epoch':epoch}
#     torch.save(state, f'{save_path}/Epoch_{epoch}_{time.strftime("%Y%m%d_%H%M%S")}.pth')
#     print(f'Epoch: {epoch:02d}, Test IoU: {iou:.4f}')

# test script
for epoch in range(1, 2):
    loss = train()
    print(f'Epoch: {epoch:02d}')
    state = {'net':model.state_dict(), 'epoch':epoch}
    torch.save(state, f'{save_path}/Epoch_{epoch}_{time.strftime("%Y%m%d_%H%M%S")}.pth')



# # debugging test()
# state_dict = torch.load('./run/Epoch_3_20221214_210321.pth')['net']
# model.load_state_dict(state_dict)
# test(test_loader, model)