import os
import time
import timeit
import yaml
import datetime
import random

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
# from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import tboard, utils
from utils.discriminative_loss import DiscriminativeLoss
# from utils.clustering import cluster # sklearn clustering
# from utils.clustering import MeanShift_GPU # GPU clustering
from utils.meanshift.mean_shift_gpu import MeanShiftEuc
from utils.eval import preprocess_pred_inst
from utils.semantic_kitti_eval_np import PanopticEval
from utils.utils import save_pred_to_disk
from utils.utils import calc_miou, calc_iou_per_cat, averaging_ious
from utils.utils import set_random_seeds
from torchmetrics.classification import MulticlassAveragePrecision

from model.PointNet2 import Net
from model.Randlanet import RandlaNet

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
                                sequences= testing_sequences, 
                                DATA_dir=DATA_path)


test_dataset = SemanticKittiGraph(dataset_dir=data_path, 
                                sequences= val_sequences, 
                                DATA_dir=DATA_path)

set_random_seeds(42)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8, persistent_workers=torch.cuda.is_available(), pin_memory=torch.cuda.is_available(),
                          drop_last=True)

# sub_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=8, persistent_workers=torch.cuda.is_available(), pin_memory=torch.cuda.is_available(),
#                           drop_last=True)
                          
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
run = str(11)
save_path = os.path.join(save_path, run) # model state path
if not os.path.exists(save_path):
    os.makedirs(save_path)
print('Run Number is :', run)

# create a SummaryWriter to write logs to a log directory
log_path = 'log_inst_loss'
log_path = os.path.join(log_path, run) # train/test info path
writer = SummaryWriter(log_dir=log_path)

# device = torch.device('cpu') # only for debugging
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net(train_dataset.get_n_classes()).to(device)
model = RandlaNet(num_features=3,
        num_classes=20,
        decimation=4,
        num_neighbors=16, 
        dim_inst_out=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # for PointNet++
# scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True) # for PointNet++
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # for Randlanet
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95) # for Randlanet
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
nloss = torch.nn.NLLLoss(weight=loss_w).to(device) # semantic branch loss
discriminative_loss = DiscriminativeLoss(delta_var=0.5, delta_dist=1.5, norm=1, alpha=1.0, beta=1.0, gamma=0.001,usegpu=True) # instance branch loss
# average_precision = MulticlassAveragePrecision(num_classes=train_dataset.get_n_classes(), average=None, thresholds=None)


def train(model, epoch, stage):
    model.train()

    # total_loss = correct_nodes = total_nodes = 0
    total_loss = 0
    sem_loss_total = 0
    inst_loss_total = 0
    count = 0
    count_max = 25

    for i, data in enumerate(train_loader):
        # Yang: train for only 100 loaders
        if count >= count_max:
            break

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
            # print('sem_loss_total', sem_loss_total)
            # print('inst_loss_total', inst_loss_total)
            
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # if (i + 1) % 50 == 0:

            # ious = calc_iou_per_cat(pred=sem_pred, target=data.x, num_classes=test_dataset.get_n_classes(), ignore_index=ignore_label)
            # miou = calc_miou(ious)
            # writer.add_scalar(f'train/miou/{stage}', miou, epoch*Length+i)

        
        count += 1
    # writer.add_scalar(f'train/{stage}/weighted_total_loss', total_loss, epoch)
    # writer.add_scalar(f'train/{stage}/weighted_inst_loss', inst_loss_total, epoch)
    # writer.add_scalar(f'train/{stage}/weighted_sem_loss', sem_loss_total, epoch)
    # writer.add_scalar(f'train/{stage}/original_inst_loss', inst_loss_total/0.7, epoch)
    return total_loss/count_max, sem_loss_total/count_max, inst_loss_total/count_max


# output iou and loss
@torch.no_grad()
def test(loader, model, if_save_pred):
    stuff_labels = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    stuff_labels = np.array(stuff_labels)
    # ic = 0 # for quick debugging
    save_idx = 0 
    count = 0
    model.eval()

    # TODO: convert the lists into a dict to improve readibility
    pq_list, sq_list, rq_list, all_pq_list, all_sq_list, all_rq_list, iou_list, all_iou_list = [], [], [], [], [], [], [], []
    
    for data in loader:
        if count >= 10:
            break

        data = data.to(device)

        sem_preds, inst_outs, seed_idx = model(data) # sem_preds size: [NpXNc], inst_outs size: [NpXNe]

        # break down for each batch
        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for sem_pred, inst_out, sem_label, inst_label, pos in zip(sem_preds.split(sizes), inst_outs.split(sizes), data.y.split(sizes), data.z.split(sizes), data.pos.split(sizes)):
            
            # take argmax of sem_pred and ignore the unlabeled category 0. sem_pred size: [NpX20] -> [1xNp]
            sem_pred[:, ignore_label] = torch.tensor([-float('inf')])
            sem_pred = sem_pred.argmax(-1)

            # inst_out_buffer = inst_out

            # normalize point features by mean and standard deviation for better clustering

            inst_out_mean = torch.mean(inst_out, dim=0)
            # print('inst_out_mean', inst_out_mean)
            inst_out_std = torch.std(inst_out, dim=0)
            # print('inst_out_std', inst_out_std)
            inst_out = (inst_out - inst_out_mean) / inst_out_std
            # print('inst_out', inst_out)

            # debugging test for using seeds from SA module and nearest on pos
            # masked_pos = pos[seed_idx]

            masked_inst_out = inst_out[seed_idx]
            meanshift = MeanShiftEuc(bandwidth=0.6)

            masked_clustering = meanshift.fit(masked_inst_out)


            masked_cluster_centers, masked_cluster_labels = masked_clustering.cluster_centers_, masked_clustering.labels_
            maksed_cluster_centers = torch.from_numpy(masked_cluster_centers).float()
            num_inst_ids = len(np.unique(masked_clustering.labels_))
            print(f'number of instance ids: {num_inst_ids}')

            # clustering = nearest(pos, masked_pos.to(device))
            # clustering = nearest(inst_out, masked_inst_out.to(device))
            clustering = nearest(inst_out, maksed_cluster_centers.to(device))
            # breakpoint()
            # clustering = nearest(pos, pos[global_idx])

            inst_pred = clustering

            # PQ evaluation
            sem_pred, inst_pred, sem_label, inst_label = preprocess_pred_inst(sem_pred, inst_pred, sem_label, inst_label) # convert to numpy unit32

            absence_label = np.setdiff1d(np.arange(test_dataset.get_n_classes()), sem_label)
            print('absence_label: ', absence_label)
            # ignore_absence_stuff_label = np.unique(np.concatenate(absence_label, np.array([ignore_label]), stuff_labels))
            ignore_absence_stuff_label = np.unique(np.concatenate([absence_label, np.array([ignore_label]), stuff_labels]))



            th_evaluator = PanopticEval(test_dataset.get_n_classes(), ignore=ignore_absence_stuff_label)
            # evaluator.addBatch(sem_pred, inst_pred, sem_label, inst_label)
            # # Yang: for debugging purpose
            # evaluator.addBatch(sem_label, inst_label, sem_label, inst_label)
            pq, sq, rq, all_pq, all_sq, all_rq = th_evaluator.getPQ()
            iou, all_iou = th_evaluator.getSemIoU() # IoU over all classes, IoU for each class

            # np.set_printoptions(threshold=np.inf)
            print(pq, sq, rq, all_pq, all_sq, all_rq, iou, all_iou)
            
            # append values into lists
            pq_list.append(pq)
            sq_list.append(sq)
            rq_list.append(rq)
            all_pq_list.append(all_pq)
            all_sq_list.append(all_sq)
            all_rq_list.append(all_rq)
            iou_list.append(iou)
            all_iou_list.append(all_iou)

            if if_save_pred == True:
                sem_pred_original = test_dataset.to_original(sem_pred) # convert from xentropy label to original label
                # data dtype should be both np.uint32 for bit shifting
                label = (inst_pred << 16) | sem_pred_original.astype(np.uint32)
                # print(label.shape,type(label))
                # print(sem_pred.shape)
                # print(inst_label.shape)
                # print(sem_pred_original.shape)
                # print(label, sem_pred_original, inst_pred) 
                save_pred_to_disk(pred=label, sequence='08', save_pred_path=prediction_path, run=run, save_idx=save_idx)
            else:
                continue

        count += 1

        save_idx += 1

        # if ic >= 1:
        #     break
        # else:
        #     ic += 1
    
    # average over test set
        # metrics
    pq_list_avr = np.mean(pq_list, axis=0)
    sq_list_avr = np.mean(sq_list, axis=0)
    rq_list_avr = np.mean(rq_list, axis=0)
    all_pq_list_avr = np.mean(all_pq_list, axis=0)
    all_sq_list_avr = np.mean(all_sq_list, axis=0)
    all_rq_list_avr = np.mean(all_rq_list, axis=0)
    iou_list_avr = np.mean(iou_list, axis=0)
    all_iou_list_avr = np.mean(all_iou_list, axis=0)
    return pq_list_avr, sq_list_avr, rq_list_avr, all_pq_list_avr, all_sq_list_avr, all_rq_list_avr, iou_list_avr, all_iou_list_avr


# output iou and loss
@torch.no_grad()
def test_feature(loader, model, if_save_pred):
    # stuff_labels = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # stuff_labels = np.array(stuff_labels)
    # ic = 0 # for quick debugging
    save_idx = 0 
    count = 0
    model.eval()

    # TODO: convert the lists into a dict to improve readibility
    pq_list, sq_list, rq_list, all_pq_list, all_sq_list, all_rq_list, iou_list, all_iou_list = [], [], [], [], [], [], [], []
    
    for data in loader:
        if count >= 25:
            break

        data = data.to(device)

        sem_preds, inst_outs, seed_idx = model(data) # sem_preds size: [NpXNc], inst_outs size: [NpXNe]

        # break down for each batch
        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for sem_pred, inst_out, sem_label, inst_label, pos in zip(sem_preds.split(sizes), inst_outs.split(sizes), data.y.split(sizes), data.z.split(sizes), data.pos.split(sizes)):
            
            # take argmax of sem_pred and ignore the unlabeled category 0. sem_pred size: [NpX20] -> [1xNp]
            sem_pred[:, ignore_label] = torch.tensor([-float('inf')])
            sem_pred = sem_pred.argmax(-1)
            print('sem_pred_0:', sem_pred)
            # inst_out_buffer = inst_out

            # normalize point features by mean and standard deviation for better clustering

            inst_out_mean = torch.mean(inst_out, dim=0)
            # print('inst_out_mean', inst_out_mean)
            inst_out_std = torch.std(inst_out, dim=0)
            # print('inst_out_std', inst_out_std)
            inst_out = (inst_out - inst_out_mean) / inst_out_std
            # print('inst_out', inst_out)

            # debugging test for using seeds from SA module and nearest on pos
            # masked_pos = pos[seed_idx]

            masked_inst_out = inst_out
            meanshift = MeanShiftEuc(bandwidth=0.6)

            masked_clustering = meanshift.fit(masked_inst_out)


            masked_cluster_centers, masked_cluster_labels = masked_clustering.cluster_centers_, masked_clustering.labels_

            num_inst_ids = len(np.unique(masked_clustering.labels_))
            print(f'number of instance ids: {num_inst_ids}')

            inst_pred = masked_cluster_labels
            # breakpoint()

            # PQ evaluation
            sem_pred, inst_pred, sem_label, inst_label = preprocess_pred_inst(sem_pred, inst_pred, sem_label, inst_label) # convert to numpy unit32
            print('sem_pred_1:', sem_pred)

            # breakpoint()

            absence_label = np.setdiff1d(np.arange(test_dataset.get_n_classes()), sem_label)
            print('absence_label: ', absence_label)
            ignore_absence_stuff_label = np.unique(np.append(absence_label, ignore_label))
            # ignore_absence_stuff_label = np.unique(np.concatenate([absence_label, np.array([ignore_label]), stuff_labels]))



            th_evaluator = PanopticEval(test_dataset.get_n_classes(), ignore=ignore_absence_stuff_label)
            th_evaluator.addBatch(sem_pred, inst_pred, sem_label, inst_label)
            # # Yang: for debugging purpose
            # th_evaluator.addBatch(sem_label, inst_label, sem_label, inst_label)
            pq, sq, rq, all_pq, all_sq, all_rq = th_evaluator.getPQ()
            iou, all_iou = th_evaluator.getSemIoU() # IoU over all classes, IoU for each class

            # np.set_printoptions(threshold=np.inf)
            print(pq, sq, rq, all_pq, all_sq, all_rq, iou, all_iou)
            
            # append values into lists
            pq_list.append(pq)
            sq_list.append(sq)
            rq_list.append(rq)
            all_pq_list.append(all_pq)
            all_sq_list.append(all_sq)
            all_rq_list.append(all_rq)
            iou_list.append(iou)
            all_iou_list.append(all_iou)

            if if_save_pred == True:
                sem_pred_original = test_dataset.to_original(sem_pred) # convert from xentropy label to original label
                # data dtype should be both np.uint32 for bit shifting
                label = (inst_pred << 16) | sem_pred_original.astype(np.uint32)
                # print(label.shape,type(label))
                # print(sem_pred.shape)
                # print(inst_label.shape)
                # print(sem_pred_original.shape)
                # print(label, sem_pred_original, inst_pred) 
                save_pred_to_disk(pred=label, sequence='08', save_pred_path=prediction_path, run=run, save_idx=save_idx)
            else:
                continue

        count += 1

        save_idx += 1

        # if ic >= 1:
        #     break
        # else:
        #     ic += 1
    
    # average over test set
        # metrics
    pq_list_avr = np.mean(pq_list, axis=0)
    sq_list_avr = np.mean(sq_list, axis=0)
    rq_list_avr = np.mean(rq_list, axis=0)
    all_pq_list_avr = np.mean(all_pq_list, axis=0)
    all_sq_list_avr = np.mean(all_sq_list, axis=0)
    all_rq_list_avr = np.mean(all_rq_list, axis=0)
    iou_list_avr = np.mean(iou_list, axis=0)
    all_iou_list_avr = np.mean(all_iou_list, axis=0)
    # breakpoint()
    return pq_list_avr, sq_list_avr, rq_list_avr, all_pq_list_avr, all_sq_list_avr, all_rq_list_avr, iou_list_avr, all_iou_list_avr


start_epoch = 0
val_count = 0
for epoch in range(start_epoch, 20):
    # Yang: for debugging Randlanet script
    stage = 'semantic'
    # stage = 'semantic'
    print(f'Start training {epoch}') 
    total_loss, sem_loss, inst_loss = train(model, epoch, stage='semantic')
    print(f'Finished training {epoch}')
    state = {'net':model.state_dict(), 'epoch':epoch, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    torch.save(state, f'{save_path}/Epoch_{epoch}_{time.strftime("%Y%m%d_%H%M%S")}_{stage}.pth')
    print(f'Finished saving model {epoch}')

    writer.add_scalar('train_loss/inst_loss', inst_loss, epoch)
    writer.add_scalar('train_loss/sem_loss', sem_loss, epoch)
    writer.add_scalar('train_loss/total_loss', total_loss, epoch)
    # validation
    # torch.load('./run_sem/3/Epoch_9_20230306_224830_semantic.pth') # this is the tenth epoch
    # model.load_state_dict(state_dict['net'])
    # ious_all, miou_all = test(test_loader, model)

    # if (val_count+1) % 10 == 0:
    #     pq_list_avr, sq_list_avr, rq_list_avr, all_pq_list_avr, all_sq_list_avr, all_rq_list_avr, iou_list_avr, all_iou_list_avr = test(test_loader, model, if_save_pred=False)
    #     writer.add_scalar('val/miou', miou_all, epoch) # the 10th epoch
    #     tboard.add_list(writer, ious_all, epoch, DATA_path, 'IoU_all')
    #     print(f'Finished tensorboarding metrics {epoch}')


    val_count += 1
    # # print(f'Finished Epoch: {epoch:02d}')
    


# # For Inference
# for epoch in range(1):
#     # testing
#     state_dict = torch.load('./run_inst_loss/10/Epoch_13_20230410_161740_semantic.pth')
#     model.load_state_dict(state_dict['net'])

#     pq_list_avr, sq_list_avr, rq_list_avr, all_pq_list_avr, all_sq_list_avr, all_rq_list_avr, iou_list_avr, all_iou_list_avr = test_feature(train_loader, model, if_save_pred=True)
#     writer.add_scalar('test/pq', pq_list_avr, epoch)
#     writer.add_scalar('test/sq', sq_list_avr, epoch)
#     writer.add_scalar('test/rq', rq_list_avr, epoch)
#     writer.add_scalar('test/mIoU', iou_list_avr, epoch) # miou

#     tboard.add_list(writer, all_pq_list_avr, epoch, DATA_path, 'PQ_all') 
#     tboard.add_list(writer, all_sq_list_avr, epoch, DATA_path, 'SQ_all')
#     tboard.add_list(writer, all_rq_list_avr, epoch, DATA_path, 'RQ_all')
#     tboard.add_list(writer, all_iou_list_avr, epoch, DATA_path, 'IoU_all')
#     print(f'Finished tensorboarding {epoch}')

# print('All finished!')