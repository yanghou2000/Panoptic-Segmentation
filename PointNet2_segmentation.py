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
from utils.discriminative_loss import DiscriminativeLoss
from utils.clustering import cluster
from utils.eval import preprocess_pred_inst
from utils.semantic_kitti_eval_np import PanopticEval

from model.PointNet2 import Net

# define the gpu index to train
os.environ["CUDA_VISIBLE_DEVICES"]="1"

data_path = '/Volumes/scratchdata/kitti/dataset/'
# DATA_path = '/home/yanghou/project/Panoptic-Segmentation/semantic-kitti.yaml' #
DATA_path = './semantic-kitti.yaml' # for running in docker
save_path = './run_inst'

testing_sequences = ['00']

train_sequences = ['00', '01', '02', '03', '04', '05', '06', '07']
val_sequences = ['08']
test_sequences = ['09', '10']

# Debugging test going on...
train_dataset = SemanticKittiGraph(dataset_dir='/Volumes/scratchdata/kitti/dataset/', 
                                sequences= testing_sequences, 
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
run = str(3)
save_path = os.path.join(save_path, run) # model state path
if not os.path.exists(save_path):
    os.makedirs(save_path)
print('Run Number is :', run)

# create a SummaryWriter to write logs to a log directory
log_path = 'log_draft'
log_path = os.path.join(log_path, run) # train/test info path
writer = SummaryWriter(log_dir=log_path, filename_suffix=time.strftime("%Y%m%d_%H%M%S"))

# device = torch.device('cpu') # only for debugging
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.get_n_classes()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
nloss = torch.nn.NLLLoss(weight=loss_w).to(device) # semantic branch loss
discriminative_loss = DiscriminativeLoss(delta_var=0.5, delta_dist=1.5,norm=1, alpha=1.0, beta=1.0, gamma=0.001,usegpu=True) # instance branch loss


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
            data_time.append(time.time() - end)
            data = data.to(device)
            optimizer.zero_grad()
            forward_start = time.time()
            sem_pred, inst_out = model(data) # inst_out size: [NpXNe]
            # print(f'inst_out.size: {inst_out.size()}', f'data.z.shape: {data.z.size()}')
            forward_time.append(time.time() - forward_start)

            # Semantic negative log likelihood loss. Input should be log-softmax
            sem_loss = nloss(sem_pred, data.y.cuda())

            # Instance discriminative loss. Input should be
            inst_loss = discriminative_loss(inst_out.permute(1, 0).unsqueeze(0), data.z.unsqueeze(0)) # input size: [NpXNe] -> [1XNeXNp] defined in discriminative loss, target size: [1XNp] -> [1X1xNp]

            # Combine semantic and instance losses together
            loss = sem_loss + inst_loss
            
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

            if (i + 1) % 50 == 0:
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
    # ious, mious = [], []
    pq_list, sq_list, rq_list, all_pq_list, all_sq_list, all_rq_list, iou_list, all_iou_list = [], [], [], [], [], [], [], []
    #TODO: debugging the test script and add instance branch evaluation 
    for data in loader:
        data = data.to(device)
        sem_preds, inst_outs = model(data) # sem_preds size: [NpXNc], inst_outs size: [NpXNe]

        # break down for each batch
        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for sem_pred, inst_out, sem_label, inst_label in zip(sem_preds.split(sizes), inst_outs.split(sizes), data.y.split(sizes), data.z.split(sizes)):
            
            # # semantic branch
            # # print('sem_pred.size: ', sem_pred.size(), 'sem_label.size: ', sem_label.size())
            # iou = utils.calc_iou_per_cat(sem_pred, sem_label, num_classes=20, ignore_index=ignore_label)
            # miou = utils.calc_miou(iou, num_classes=20, ignore_label=True)
            # # print('iou: ', iou, 'miou: ', miou)
            # ious.append(iou)
            # mious.append(miou)

            # take argmax of sem_pred and ignore the unlabeled category 0. sem_pred size: [NpX20] -> [1xNp]
            sem_pred[:, ignore_label] = torch.tensor([-float('inf')])
            sem_pred = sem_pred.argmax(-1)

            # normalize point features by mean and standard deviation for better clustering
            inst_out_mean = torch.mean(inst_out, dim=0)
            print('inst_out_mean', inst_out_mean)
            inst_out_std = torch.std(inst_out, dim=0)
            print('inst_out_std', inst_out_std)
            inst_out = (inst_out - inst_out_mean) / inst_out_std
            print('inst_out', inst_out)

            # # instance branch
            inst_num_clusters, inst_pred, inst_cluster_centers = cluster(inst_out, bandwidth=2.0, n_jobs=-1) # dtype=torch
            
            # to device
            inst_pred = inst_pred.to(device)

            # PQ evaluation
            print('sem_pred.size', sem_pred.size())
            print('inst_pred.size', inst_pred.size())
            print('sem_label.size', sem_label.size())
            print('inst_label.size', inst_label.size())
            sem_pred, inst_pred, sem_label, inst_label = preprocess_pred_inst(sem_pred, inst_pred, sem_label, inst_label) # convert to numpy int 
            evaluator = PanopticEval(20, ignore=[ignore_label])
            evaluator.addBatch(sem_pred, inst_pred, sem_label, inst_label)
            pq, sq, rq, all_pq, all_sq, all_rq = evaluator.getPQ()
            iou, all_iou = evaluator.getSemIoU() # IoU over all classes, IoU for each class

            # append values into lists
            pq_list.append(pq)
            sq_list.append(sq)
            rq_list.append(rq)
            all_pq_list.append(all_pq)
            all_sq_list.append(all_sq)
            all_rq_list.append(all_rq)
            iou_list.append(iou)
            all_iou_list.append(all_iou)


    # # process mious and ious into right format
    # ious = torch.cat(ious, dim=-1).reshape(-1, 20) # concatenate into a tensor of tensors
    # mious = torch.as_tensor(mious) # convert list into view of tensor

    # # average over test set
    # ious_avr = utils.averaging_ious(ious) # record the number of nan in each array, filled with 0, take sum, divied by (num_classes-number of nan)
    # miou_avr = torch.sum(mious) / mious.size()[0]
    
    # average over test set
    pq_list_avr = np.mean(pq_list, axis=1)
    sq_list_avr = np.mean(pq_list, axis=1)
    rq_list_avr = np.mean(rq_list, axis=1)
    all_pq_list_avr = np.mean(all_pq_list, axis=0)
    all_sq_list_avr = np.mean(all_sq_list, axis=0)
    all_rq_list_avr = np.mean(all_rq_list, axis=0)
    iou_list_avr = np.mean(iou_list, axis=1)
    all_iou_list_avr = np.mean(all_iou_list, axis=0)

    return pq_list_avr, sq_list_avr, rq_list_avr, all_pq_list_avr, all_sq_list_avr, all_rq_list_avr, iou_list_avr, all_iou_list_avr


for epoch in range(11): # original is (1, 31)
    train(epoch)
    print(f'Finished training {epoch}')
    state = {'net':model.state_dict(), 'epoch':epoch, 'optimizer': optimizer}
    torch.save(state, f'{save_path}/Epoch_{epoch}_{time.strftime("%Y%m%d_%H%M%S")}.pth')
    print(f'Finished saving model {epoch}')

    # # debugging test by loading model
    # state_dict = torch.load('./run_inst/2/Epoch_0_20230211_182726.pth')['net']
    # model.load_state_dict(state_dict)

    # ious_all, miou_all = test(test_loader, model) # metrics over the whole test set
    pq_list, sq_list, rq_list, all_pq_list, all_sq_list, all_rq_list, iou_list, all_iou_list = test(test_loader, model)
    print(f'Finished testing {epoch}')

    # tensorboarding
    writer.add_scalar('test/pq', pq_list, epoch)
    writer.add_scalar('test/sq', sq_list, epoch)
    writer.add_scalar('test/rq', rq_list, epoch)
    writer.add_scalar('test/miou', iou_list, epoch) # miou

    tboard.add_list(writer, all_pq_list, epoch, DATA_path) 
    tboard.add_list(writer, all_sq_list, epoch, DATA_path)
    tboard.add_list(writer, all_rq_list, epoch, DATA_path)
    tboard.add_list(writer, all_iou_list, epoch, DATA_path)
    print(f'Finished tensorboarding {epoch}')

    print(f'Finished Epoch: {epoch:02d}')
    
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