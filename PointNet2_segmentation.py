import os
import time
import timeit
import yaml
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from Modules import GlobalSAModule, SAModule
from torch_geometric.nn import fps
import torch.profiler
from torch_cluster import fps
from torch_cluster import nearest

import torch_geometric.transforms as T
from Semantic_Dataloader import SemanticKittiGraph
from torch_geometric.loader import DataLoader

from torch_geometric.data import Dataset, Data
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import tboard, utils
from utils.discriminative_loss import DiscriminativeLoss
# from utils.clustering import cluster # sklearn clustering
# from utils.clustering import MeanShift_GPU # GPU clustering
from utils.meanshift.mean_shift_gpu import MeanShiftEuc
from utils.eval import preprocess_pred_inst
from utils.semantic_kitti_eval_np import PanopticEval
from utils.utils import calc_miou, calc_iou_per_cat, averaging_ious

from model.PointNet2 import Net
from model.Randlanet import RandlaNet

# define the gpu index to train
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# data_path = '/Volumes/scratchdata/kitti/dataset/'
# data_path = '/Volumes/scratchdata_smb/kitti/dataset/' # alternative path
data_path = '/Volumes/mrgdatastore6/ThirdPartyData/semantic_kitti/dataset' # alternative path, if not startwith ._ in dataloader and in visualizer
# DATA_path = '/home/yanghou/project/Panoptic-Segmentation/semantic-kitti.yaml' #
DATA_path = './semantic-kitti.yaml' # for running in docker
save_path = './run_sem'
prediction_path = './panoptic_data'

testing_sequences = ['00']

train_sequences = ['00', '01', '02', '03', '04', '05', '06', '07']
val_sequences = ['08']
test_sequences = ['09', '10']

# Debugging test going on...
train_dataset = SemanticKittiGraph(dataset_dir=data_path, 
                                sequences= train_sequences, 
                                DATA_dir=DATA_path)

test_dataset = SemanticKittiGraph(dataset_dir=data_path, 
                                sequences= val_sequences, 
                                DATA_dir=DATA_path)

torch.manual_seed(42)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, persistent_workers=torch.cuda.is_available(), pin_memory=torch.cuda.is_available(),
                          drop_last=True)
# Yang: changed the batch size from 8 to 2 for degbugging the model
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, persistent_workers=torch.cuda.is_available(), pin_memory=torch.cuda.is_available(),
                         drop_last=True)

# add ignore index for ignore labels in training and testing
ignore_label = 0

# add loss weights beforehand
loss_w = train_dataset.map_loss_weight()
loss_w[ignore_label] = 0 # set the label to be zero so no training for this category
print('loss_w, check first element to be zero ', loss_w)

# make run file, update for every run
run = str(6)
save_path = os.path.join(save_path, run) # model state path
if not os.path.exists(save_path):
    os.makedirs(save_path)
print('Run Number is :', run)

# create a SummaryWriter to write logs to a log directory
log_path = 'log_sem'
log_path = os.path.join(log_path, run) # train/test info path
writer = SummaryWriter(log_dir=log_path, filename_suffix=time.strftime("%Y%m%d_%H%M%S"))

# device = torch.device('cpu') # only for debugging
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net(train_dataset.get_n_classes()).to(device)
model = RandlaNet(num_features=3,
        num_classes=20,
        decimation=4,
        num_neighbors=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # for PointNet++
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True) # for PointNet++
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # for Randlanet
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95) # for Randlanet
nloss = torch.nn.NLLLoss(weight=loss_w).to(device) # semantic branch loss
discriminative_loss = DiscriminativeLoss(delta_var=0.5, delta_dist=1.5, norm=1, alpha=1.0, beta=1.0, gamma=0.001,usegpu=True) # instance branch loss


def train(epoch, stage):
    model.train()

    # total_loss = correct_nodes = total_nodes = 0
    total_loss = 0
    sem_loss_total = 0
    inst_loss_total = 0
    Length = len(train_loader)

    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        sem_pred, inst_out, _ = model(data) # inst_out size: [NpXNe]

        # print(f'inst_out.size: {inst_out.size()}', f'data.z.shape: {data.z.size()}')
        if stage == 'semantic':
            # Semantic negative log likelihood loss. Input should be log-softmax
            sem_loss = nloss(sem_pred, data.y.cuda())
            loss = sem_loss
            sem_loss_total += sem_loss.item()

        else:
            sem_loss = nloss(sem_pred, data.y.cuda())

            # Instance discriminative loss. Input should be
            inst_loss = discriminative_loss(inst_out.permute(1, 0).unsqueeze(0), data.z.unsqueeze(0)) # input size: [NpXNe] -> [1XNeXNp] defined in discriminative loss, target size: [1XNp] -> [1X1xNp]

            # Combine semantic and instance losses together
            '''PointNet++''' 
            # loss = 0.13*sem_loss + 0.87*inst_loss
            # sem_loss_total += 0.13*sem_loss.item()
            # inst_loss_total += 0.87*inst_loss.item()
            
            '''Randlanet'''
            loss = 0.45*sem_loss + 0.55*inst_loss
            sem_loss_total += 0.45*sem_loss.item()
            inst_loss_total += 0.55*inst_loss.item()
            
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 50 == 0:
            writer.add_scalar(f'train/{stage}/total_loss', total_loss, epoch*Length+i)
            writer.add_scalar(f'train/{stage}/inst_loss', inst_loss_total, epoch*Length+i)
            writer.add_scalar(f'train/{stage}/sem_loss', sem_loss_total, epoch*Length+i)
            # ious = calc_iou_per_cat(pred=sem_pred, target=data.x, num_classes=test_dataset.get_n_classes(), ignore_index=ignore_label)
            # miou = calc_miou(ious)
            # writer.add_scalar(f'train/miou/{stage}', miou, epoch*Length+i)

            total_loss = 0
            sem_loss_total = 0
            inst_loss_total = 0
        
@torch.no_grad()
def test(loader, model):
    # ic = 0 # for quick debugging
    model.eval()
    ious, mious = [], []
    for data in loader:
            data = data.to(device)
            sem_preds, _, _ = model(data)
            sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
            for sem_pred, sem_label in zip(sem_preds.split(sizes), data.y.split(sizes)):
                iou = calc_iou_per_cat(pred=sem_pred, target=sem_label, num_classes=test_dataset.get_n_classes(), ignore_index=ignore_label)
                miou = calc_miou(iou, num_classes=test_dataset.get_n_classes(), ignore_label=True)

                print('iou: ', iou, 'miou: ', miou)
                ious.append(iou)
                mious.append(miou)
    ious = torch.cat(ious, dim=-1).reshape(-1, 20) # concatenate into a tensor of tensors
    mious = torch.as_tensor(mious) # convert list into view of tensor

    ious_avr = utils.averaging_ious(ious) # record the number of nan in each array, filled with 0, take sum, divied by (num_classes-number of nan)
    miou_avr = torch.sum(mious) / mious.size()[0]
    return ious_avr, miou_avr


# TODO: add arguments so to better resume training 
# resume training
state_dict = torch.load('./run_sem/6/Epoch_14_20230403_075049_semantic_instance.pth')
model.load_state_dict(state_dict['net'])
optimizer.load_state_dict(state_dict['optimizer'])
# reduce the learning rate by 10 as it starts after 10 epochs
# optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10 # only for hacking, after adding schduler.step, this line is not needed
print('learning rate used is: ', optimizer.param_groups[0]['lr'])
start_epoch = state_dict['epoch'] + 1
start_epoch = 15

for epoch in range(start_epoch, 60):
    # Yang: for debugging Randlanet script
    stage = 'semantic_instance'
    # stage = 'semantic'
    print(f'Start training {epoch}') 
    train(epoch, stage=stage)
    print(f'Finished training {epoch}')
    state = {'net':model.state_dict(), 'epoch':epoch, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    torch.save(state, f'{save_path}/Epoch_{epoch}_{time.strftime("%Y%m%d_%H%M%S")}_{stage}.pth')
    print(f'Finished saving model {epoch}')

    # validation
    # torch.load('./run_sem/3/Epoch_9_20230306_224830_semantic.pth') # this is the tenth epoch
    # model.load_state_dict(state_dict['net'])
    ious_all, miou_all = test(test_loader, model)

    writer.add_scalar('val/miou', miou_all, epoch) # the 10th epoch
    tboard.add_list(writer, ious_all, epoch, DATA_path, 'IoU_all')
    print(f'Finished tensorboarding metrics {epoch}')

    scheduler.step(miou_all)
    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
    print(f'Finished tensorboarding learning rate {epoch}')


    # # print(f'Finished Epoch: {epoch:02d}')
    

print('All finished!')
    

# # resume testing
# epoch = 10
# state_dict = torch.load('./run_sem/4/Epoch_10_20230309_160615_semantic_instance.pth')
# model.load_state_dict(state_dict['net'])
# ious_all, miou_all = test(test_loader, model)
# tboard.add_list(writer, ious_all, epoch, DATA_path, 'test/iou')
# print(f'Finished tensorboarding metrics {epoch}')
# writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
# print(f'Finished tensorboarding learning rate {epoch}')



