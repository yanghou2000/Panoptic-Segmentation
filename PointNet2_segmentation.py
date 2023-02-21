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

from utils import tboard, utils
from utils.discriminative_loss import DiscriminativeLoss
# from utils.clustering import cluster # sklearn clustering
# from utils.clustering import MeanShift_GPU # GPU clustering
from utils.meanshift.mean_shift_gpu import MeanShiftEuc
from utils.eval import preprocess_pred_inst
from utils.semantic_kitti_eval_np import PanopticEval
from utils.utils import save_tensor_to_disk

from model.PointNet2 import Net

# define the gpu index to train
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# data_path = '/Volumes/scratchdata/kitti/dataset/'
data_path = '/Volumes/scratchdata_smb/kitti/dataset/' # alternative path
# DATA_path = '/home/yanghou/project/Panoptic-Segmentation/semantic-kitti.yaml' #
DATA_path = './semantic-kitti.yaml' # for running in docker
save_path = './run_inst'
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

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=8, persistent_workers=torch.cuda.is_available(), pin_memory=torch.cuda.is_available(),
                          drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8, persistent_workers=torch.cuda.is_available(), pin_memory=torch.cuda.is_available(),
                         drop_last=True)

# add ignore index for ignore labels in training and testing
ignore_label = 0

# add loss weights beforehand
loss_w = train_dataset.map_loss_weight()
loss_w[ignore_label] = 0 # set the label to be zero so no training for this category
print('loss_w, check first element to be zero ', loss_w)

# make run file, update for every run
run = str(5)
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
            
            total_time.append(time.time() - end)

            if (i + 1) % 50 == 0:
                writer.add_scalar('train/total_loss', total_loss, epoch*Length+i)
                total_loss = 0
                data_time, forward_time, total_time = [], [], []
            
            prof.step()
            end = time.time()

        


# output iou and loss
@torch.no_grad()
def test(loader, model):
    # ic = 0 # for quick debugging

    model.eval()

    pq_list, sq_list, rq_list, all_pq_list, all_sq_list, all_rq_list, iou_list, all_iou_list = [], [], [], [], [], [], [], []
    
    #TODO: add timing
    # time_fps_list, time_norm_list, time_cluster_list, time_nearest_list = [],[],[],[]
    # all_time_list = [] # append every time category list into one list
    
    #TODO: debugging the test script and add instance branch evaluation 
    for data in loader:
        data = data.to(device)
        sem_preds, inst_outs = model(data) # sem_preds size: [NpXNc], inst_outs size: [NpXNe]

        # break down for each batch
        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for sem_pred, inst_out, sem_label, inst_label in zip(sem_preds.split(sizes), inst_outs.split(sizes), data.y.split(sizes), data.z.split(sizes)):
            
            # take argmax of sem_pred and ignore the unlabeled category 0. sem_pred size: [NpX20] -> [1xNp]
            sem_pred[:, ignore_label] = torch.tensor([-float('inf')])
            sem_pred = sem_pred.argmax(-1)

            # normalize point features by mean and standard deviation for better clustering
            torch.cuda.synchronize()
            start_time = time.time() 

            inst_out_mean = torch.mean(inst_out, dim=0)
            # print('inst_out_mean', inst_out_mean)
            inst_out_std = torch.std(inst_out, dim=0)
            # print('inst_out_std', inst_out_std)
            inst_out = (inst_out - inst_out_mean) / inst_out_std
            # print('inst_out', inst_out)

            torch.cuda.synchronize()
            end_time = time.time()
            print(f'normalization took {end_time-start_time} seconds')

            # TODO: use fps to downsample point clouds and use nearest to repropogate labels
            torch.cuda.synchronize()
            start_time = time.time()

            inst_out_fps_idx = fps(inst_out, ratio=0.1)

            torch.cuda.synchronize()
            end_time = time.time()
            print(f'fps took {end_time-start_time} seconds')

            inst_out_buffer = inst_out # save for later propogation
            inst_out = inst_out[inst_out_fps_idx]

            # # instance branch
            # inst_num_clusters, inst_pred, inst_cluster_centers = cluster(inst_out, bandwidth=2.0, n_jobs=-1) # dtype=torch
            meanshift = MeanShiftEuc(bandwidth=0.6)
            torch.cuda.synchronize()
            start_time = time.time() 

            clustering_fps = meanshift.fit(inst_out)

            torch.cuda.synchronize()
            end_time = time.time()
            print(f'meanshift took {end_time-start_time} seconds')

            # get fps-downsampled cluster centers
            cluster_centers_fps = clustering_fps.cluster_centers_
            cluster_centers_fps = torch.from_numpy(cluster_centers_fps).float()

            torch.cuda.synchronize()
            start_time = time.time() 

            clustering = nearest(inst_out_buffer, cluster_centers_fps.to(device))

            torch.cuda.synchronize()
            end_time = time.time()
            print(f'find nearest neighbors took {end_time-start_time} seconds')

            # inst_pred, _ = clustering.labels_, clustering.cluster_centers_ # type(inst_pred)=numpy.ndarray
            inst_pred = clustering # dtype = tensor
            
            # # debugging test
            # inst_pred = inst_label

            # PQ evaluation
            sem_pred, inst_pred, sem_label, inst_label = preprocess_pred_inst(sem_pred, inst_pred, sem_label, inst_label) # convert to numpy int 
            evaluator = PanopticEval(20, ignore=[ignore_label])
            # breakpoint()
            evaluator.addBatch(sem_pred, inst_pred, sem_label, inst_label)
            pq, sq, rq, all_pq, all_sq, all_rq = evaluator.getPQ()
            iou, all_iou = evaluator.getSemIoU() # IoU over all classes, IoU for each class
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

            #TODO: save predictions to disk for visualization, dtype=numpy int, concatenate the inst and pred labels together into binary format
            # save_tensor_to_disk(sem_pred, 'sem_pred.bin', prediction_path, run, sequence='08')

        # if ic >= 1:
        #     break
        # else:
        #     ic += 1


    # # process mious and ious into right format
    # ious = torch.cat(ious, dim=-1).reshape(-1, 20) # concatenate into a tensor of tensors
    # mious = torch.as_tensor(mious) # convert list into view of tensor

    # # average over test set
    # ious_avr = utils.averaging_ious(ious) # record the number of nan in each array, filled with 0, take sum, divied by (num_classes-number of nan)
    # miou_avr = torch.sum(mious) / mious.size()[0]
    
    # average over test set
    pq_list_avr = np.mean(pq_list, axis=0)
    sq_list_avr = np.mean(pq_list, axis=0)
    rq_list_avr = np.mean(rq_list, axis=0)
    all_pq_list_avr = np.mean(all_pq_list, axis=1)
    all_sq_list_avr = np.mean(all_sq_list, axis=1)
    all_rq_list_avr = np.mean(all_rq_list, axis=1)
    iou_list_avr = np.mean(iou_list, axis=0)
    all_iou_list_avr = np.mean(all_iou_list, axis=1)

    return pq_list_avr, sq_list_avr, rq_list_avr, all_pq_list_avr, all_sq_list_avr, all_rq_list_avr, iou_list_avr, all_iou_list_avr

# Yang: change from 11 to 1 for evaluting metrics
for epoch in range(1): # original is (1, 31)
    # train(epoch)
    # print(f'Finished training {epoch}')
    # state = {'net':model.state_dict(), 'epoch':epoch, 'optimizer': optimizer}
    # torch.save(state, f'{save_path}/Epoch_{epoch}_{time.strftime("%Y%m%d_%H%M%S")}.pth')
    # print(f'Finished saving model {epoch}')

    # debugging test by loading model
    state_dict = torch.load('./run_inst/5/Epoch_10_20230218_120949.pth')['net']
    model.load_state_dict(state_dict)
    
    pq_list, sq_list, rq_list, all_pq_list, all_sq_list, all_rq_list, iou_list, all_iou_list = test(test_loader, model)
    # pq_list, sq_list, rq_list, all_pq_list, all_sq_list, all_rq_list, iou_list, all_iou_list = 0.050956726561817565, 0.07214042867747611, 0.07692307692307693, np.random.rand(20).astype(float), np.random.rand(20).astype(float), np.random.rand(20).astype(float), 0.05, np.random.rand(20).astype(float)
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