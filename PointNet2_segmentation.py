import os
import time
import timeit
import yaml
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from Modules import GlobalSAModule, SAModule
from torch_scatter import scatter
import torch.profiler

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

testing_sequences = ['00']

train_sequences = ['00', '01', '02', '03', '04', '05', '06']
val_sequences = ['07']
test_sequences = ['08', '09', '10']

train_dataset = SemanticKittiGraph(dataset_dir='/Volumes/scratchdata/kitti/dataset/', 
                                sequences= train_sequences, 
                                DATA_dir=DATA_path)

test_dataset = SemanticKittiGraph(dataset_dir='/Volumes/scratchdata/kitti/dataset/', 
                                sequences= val_sequences, 
                                DATA_dir=DATA_path)

torch.manual_seed(42)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, persistent_workers=torch.cuda.is_available(), pin_memory=torch.cuda.is_available(),
                          drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=8, persistent_workers=torch.cuda.is_available(), pin_memory=torch.cuda.is_available(),
                         drop_last=True)

# add ignore index for ignore labels in training and testing
ignore_label = 0

# add loss weights beforehand
loss_w = train_dataset.map_loss_weight()
loss_w[ignore_label] = 0 # set the label to be zero so no training for this category
print('loss_w, check first element to be zero ', loss_w)

# make run file, update for every run
run = str(8)
save_path = os.path.join(save_path, run) # model state path
if not os.path.exists(save_path):
    os.makedirs(save_path)
print('Run Number is :', run)

# create a SummaryWriter to write logs to a log directory
log_path = 'tmp_log'
log_path = os.path.join(log_path, run) # train/test info path
writer = SummaryWriter(log_dir=log_path, filename_suffix=time.strftime("%Y%m%d_%H%M%S"))

# device = torch.device('cpu') # only for debugging
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.get_n_classes()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
nloss = torch.nn.NLLLoss(weight=loss_w).to(device)

# TODO: ignore class label 0
# TODO: modify train acc metric after igonoring class label 0
def train(epoch):
    model.train()

    # total_loss = correct_nodes = total_nodes = 0
    total_loss = 0
    Length = len(train_loader)

    data_time, forward_time, total_time = [], [], []
    end = time.time()
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=4),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(log_path, time.strftime("%Y%m%d_%H%M%S"))),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for i, data in enumerate(train_loader):
            # check start time
            # start = timeit.default_timer()
            data_time.append(time.time() - end)
            data = data.to(device)
            optimizer.zero_grad()
            forward_start = time.time()
            out = model(data)
            forward_time.append(time.time() - forward_start)
            # negative log likelihood loss. Input should be log-softmax
            loss = nloss(out, data.y.cuda())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
            # total_nodes += data.num_nodes
            # print('Loss: ', loss)
            
            # check end time
            # end = timeit.default_timer()
            # print('Batch_ID: ', i, 'Eplapsed time: ', end-start)
            
            total_time.append(time.time() - end)

            if (i + 1) % 10 == 0:
                # print(f'[{i+1}/{len(train_loader)}] Loss: {total_loss / 50:.4f} '
                #       f'Train Acc: {correct_nodes / total_nodes:.4f}')
                # print(f'[{i+1}/{Length}] Loss: {total_loss / 10:.4f} '
                    #   f'Train Acc: {correct_nodes / total_nodes:.4f} '
                #       f'Data Time: {datetime.timedelta(seconds=np.array(data_time).mean())} '
                #       f'Forward Time: {datetime.timedelta(seconds=np.array(forward_time).mean())} '
                #       f'Total Time: {datetime.timedelta(seconds=np.array(total_time).mean())}')
                writer.add_scalar('train/total_loss', total_loss, epoch*Length+i)
                # writer.add_scalar('train/accuracy', correct_nodes / total_nodes, epoch*Length+i)
                # total_loss = correct_nodes = total_nodes = 0
                total_loss = 0
                data_time, forward_time, total_time = [], [], []
            
            prof.step()
            end = time.time()

        


# output iou and loss
@torch.no_grad()
def test(loader, model):

    model.eval()

    # list to store the metrics for every batch
    ious, mious = [], []
    y_map = torch.empty(loader.dataset.get_n_classes(), device=device).long()
    #TODO: change the 20 into a variable. Currently the 20 categories inlcude the unlabled class which
    #should be removed. After removal there should be 19
    # data_category = list(range(20))
    for data in loader:
        data = data.to(device)
        outs = model(data)

        # Break down for each batch
        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for out, y in zip(outs.split(sizes), data.y.split(sizes)):

            print('out.size: ', out.size(), 'y.size: ', y.size())
            iou = utils.calc_iou_per_cat(out, y, num_classes=20, ignore_index=ignore_label)

            miou = utils.calc_miou(iou, num_classes=20, ignore_label=True)

            print('iou: ', iou, 'miou: ', miou)
            ious.append(iou)
            mious.append(miou)

    # process mious and ious into right format
    ious = torch.cat(ious, dim=-1).reshape(-1, 20) # concatenate into a tensor of tensors
    mious = torch.as_tensor(mious) # convert list into view of tensor

    # average over test set
    ious_avr = utils.averaging_ious(ious) # record the number of nan in each array, filled with 0, take sum, divied by (num_classes-number of nan)
    miou_avr = torch.sum(mious) / mious.size()[0]

    return ious_avr, miou_avr


for epoch in range(11): # original is (1, 31)
    train(epoch)
    print(f'Finished training {epoch}')
    state = {'net':model.state_dict(), 'epoch':epoch, 'optimizer': optimizer}
    torch.save(state, f'{save_path}/Epoch_{epoch}_{time.strftime("%Y%m%d_%H%M%S")}.pth')
    print(f'Finished saving model {epoch}')

    ious_all, miou_all = test(test_loader, model) # metrics over the whole test set
    print(f'Finished testing {epoch}')

    writer.add_scalar('test/miou', miou_all, epoch) # miou
    tboard.add_iou(writer, ious_all, epoch, DATA_path) # iou for each category
    print(f'Finished tensorboarding {epoch}')

    print(f'Finished Epoch: {epoch:02d}, Test IoU: {miou_all:.4f}')
    
print('All finished!')
    

# test script
# for epoch in range(1, 2):
#     loss = train()
#     print(f'Epoch: {epoch:02d}')
#     state = {'net':model.state_dict(), 'epoch':epoch}
#     torch.save(state, f'{save_path}/Epoch_{epoch}_{time.strftime("%Y%m%d_%H%M%S")}.pth')



# # debugging test()
# state_dict = torch.load('./run/0/Epoch_10_20221217_121731.pth')['net']
# model.load_state_dict(state_dict)
# ious_all, miou_all = test(test_loader, model)
# print('ious_all: \n', ious_all)
# print('miou_all: \n', miou_all)
# print('Finished testing')