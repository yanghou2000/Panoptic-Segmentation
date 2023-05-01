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
from torch_geometric.nn import MLP

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
from utils.utils import set_random_seeds

from model.PointNet2 import Net
from model.Randlanet import RandlaNet
from model.Randlanet_mlp import RandlaNet_mlp

# define the gpu index to train
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# data_path = '/Volumes/scratchdata/kitti/dataset/'
# data_path = '/Volumes/scratchdata_smb/kitti/dataset/' # alternative path
data_path = '/Volumes/mrgdatastore6/ThirdPartyData/semantic_kitti/dataset' # alternative path, if not startwith ._ in dataloader and in visualizer
# DATA_path = '/home/yanghou/project/Panoptic-Segmentation/semantic-kitti.yaml' #
DATA_path = './semantic-kitti.yaml' # for running in docker
save_path = './run_inst_loss'
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

set_random_seeds(42)

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
run = str(51)
save_path = os.path.join(save_path, run) # model state path
if not os.path.exists(save_path):
    os.makedirs(save_path)
print('Run Number is :', run)

# create a SummaryWriter to write logs to a log directory
log_path = 'log_inst_loss'
log_path = os.path.join(log_path, run) # train/test info path
writer = SummaryWriter(log_dir=log_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net(train_dataset.get_n_classes()).to(device)
asis = {'feature_dim': 32, 'inst_emb_dim': 5, 'num_class': 20, 'k':30, 'distance_threshold': 0.5, 'norm_degree': 2}
model = RandlaNet_mlp(num_features=3,
        num_classes=20,
        decimation=4,
        num_neighbors=16, 
        dim_inst_out=5, 
        use_asis=True,
        asis=asis).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # for PointNet++
# scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True) # for PointNet++
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # for Randlanet
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0) # for Randlanet
nloss = torch.nn.NLLLoss(weight=loss_w).to(device) # semantic branch loss
discriminative_loss = DiscriminativeLoss(delta_var=0.5, delta_dist=1.5, norm=2, alpha=1, beta=1, gamma=0.001,usegpu=True) # instance branch loss


def train(model, epoch, stage):
    model.train()

    # total_loss = correct_nodes = total_nodes = 0
    total_loss = 0
    sem_loss_total = 0
    inst_loss_total = 0
    count = 50
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
            loss = 0.3*sem_loss + 0.7*inst_loss
            sem_loss_total += 0.3*sem_loss.item()
            inst_loss_total += 0.7*inst_loss.item()
            
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (i + 1) % count == 0:
            # normalize by 50 to get loss per batch
            writer.add_scalar(f'train/{stage}/total_loss', total_loss/count, epoch*Length+i)
            writer.add_scalar(f'train/{stage}/inst_loss', inst_loss_total/count, epoch*Length+i)
            writer.add_scalar(f'train/{stage}/sem_loss', sem_loss_total/count, epoch*Length+i)
            # ious = calc_iou_per_cat(pred=sem_pred, target=data.x, num_classes=test_dataset.get_n_classes(), ignore_index=ignore_label)
            # miou = calc_miou(ious)
            # writer.add_scalar(f'train/miou/{stage}', miou, epoch*Length+i)

            total_loss = 0
            sem_loss_total = 0
            inst_loss_total = 0
        
@torch.no_grad()
def test_feature(loader, model):
    ic = 0 # for quick debugging
    model.eval()

    # TODO: convert the lists into a dict to improve readibility
    iou_list, all_iou_list = [], []
    buffer_input = torch.tensor(0)
    
    for data in loader:

        data = data.to(device)

        sem_preds, inst_outs, seed_idx = model(data) # sem_preds size: [NpXNc], inst_outs size: [NpXNe]

        # break down for each batch
        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for sem_pred, sem_label in zip(sem_preds.split(sizes), data.y.split(sizes)):
            
            # take argmax of sem_pred and ignore the unlabeled category 0. sem_pred size: [NpX20] -> [1xNp]
            sem_pred[:, ignore_label] = torch.tensor([-float('inf')])
            sem_pred_unmasked = sem_pred.argmax(-1) # [1XNp]
            unlabel_mask = torch.isin(sem_label, torch.tensor(ignore_label).to(device)) 
            sem_pred = sem_pred_unmasked.masked_fill(unlabel_mask, 0) # exclude value that is in ignore_label

            # PQ evaluation
            sem_pred, _, sem_label, _ = preprocess_pred_inst(sem_pred, buffer_input, sem_label, buffer_input) # convert to numpy unit32

            absence_label = np.setdiff1d(np.arange(test_dataset.get_n_classes()), sem_label)
            ignore_absence_stuff_label = np.unique(np.append(absence_label, ignore_label))

            evaluator = PanopticEval(test_dataset.get_n_classes(), ignore=ignore_absence_stuff_label)
            evaluator.addBatch(sem_pred, sem_label, sem_label, sem_label) # second and fourth inputs are dummy inputs for instance segmentation

            iou, all_iou = evaluator.getSemIoU() # IoU over all classes, IoU for each class
            
            iou_list.append(iou)
            all_iou_list.append(all_iou)


        # if ic >= 1:
        #     break
        # else:
        #     ic += 1
    
    # average over test set
    iou_list_avr = np.mean(iou_list, axis=0)
    all_iou_list_avr = np.mean(all_iou_list, axis=0)
    # breakpoint()
    return iou_list_avr, all_iou_list_avr


# TODO: add arguments so to better resume training 
# resume training
state_dict = torch.load('./run_inst_loss/39/Epoch_49_20230424_234534_semantic.pth')
# state_dict['net'].asis.inst_fc = MLP([8, 32], plain_last=False)  # For hacking
# nn.init.xavier_uniform_(state_dict['net'].asis.inst_fc.weight)
# nn.init.zeros_(state_dict['net'].asis.inst_fc.bias)
# state_dict['net']['asis.inst_fc.weight'] = state_dict['net'].asis.inst_fc.weight
# state_dict['net']['asis.inst_fc.bias'] = state_dict['net'].asis.inst_fc.bias

model.load_state_dict(state_dict['net'], strict=False)
# optimizer.load_state_dict(state_dict['optimizer'])
# # reduce the learning rate by 10 as it starts after 10 epochs
# # optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10 # only for hacking, after adding schduler.step, this line is not needed
# print('learning rate used is: ', optimizer.param_groups[0]['lr'])
start_epoch = state_dict['epoch'] + 1
# start_epoch = 0

for epoch in range(start_epoch, 100):
    # Yang: for debugging Randlanet script
    stage = 'semantic_instance'
    # stage = 'semantic'
    print(f'Start training {epoch}') 
    train(model, epoch, stage=stage)
    print(f'Finished training {epoch}')
    if (epoch+1) % 5 == 0:
        state = {'net':model.state_dict(), 'epoch':epoch, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
        torch.save(state, f'{save_path}/Epoch_{epoch}_{time.strftime("%Y%m%d_%H%M%S")}_{stage}.pth')
        print(f'Finished saving model {epoch}')

        ious_all, miou_all = test_feature(test_loader, model)
        writer.add_scalar('val/miou', ious_all, epoch) # the 10th epoch
        tboard.add_list(writer, miou_all, epoch, DATA_path, 'val/IoU_all')
        print(f'Finished tensorboarding metrics {epoch}')

        
        
    # scheduler.step(miou_all)
    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
    print(f'Finished tensorboarding learning rate {epoch}')


    # # print(f'Finished Epoch: {epoch:02d}')
    

print('All finished!')
    
