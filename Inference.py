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
from utils.utils import save_pred_to_disk

from model.PointNet2 import Net
from model.Randlanet import RandlaNet

# define the gpu index to train
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# data_path = '/Volumes/scratchdata/kitti/dataset/'
# data_path = '/Volumes/scratchdata_smb/kitti/dataset/' # alternative path
data_path = '/Volumes/mrgdatastore6/ThirdPartyData/semantic_kitti/dataset'
# DATA_path = '/home/yanghou/project/Panoptic-Segmentation/semantic-kitti.yaml' #
DATA_path = './semantic-kitti.yaml' # for running in docker
save_path = './run_sem'
prediction_path = './panoptic_data'

testing_sequences = ['00']

train_sequences = ['00', '01', '02', '03', '04', '05', '06', '07']
val_sequences = ['08']
test_sequences = ['09', '10']

test_dataset = SemanticKittiGraph(dataset_dir=data_path, 
                                sequences= val_sequences, 
                                DATA_dir=DATA_path)

torch.manual_seed(42)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, persistent_workers=torch.cuda.is_available(), pin_memory=torch.cuda.is_available(),
                         drop_last=True)

# add ignore index for ignore labels in training and testing
ignore_label = 0

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
# model = Net(test_dataset.get_n_classes()).to(device)
model = RandlaNet(num_features=3,
        num_classes=20,
        decimation=4,
        num_neighbors=16).to(device)

# output iou and loss
@torch.no_grad()
def test(loader, model, if_save_pred):
    # ic = 0 # for quick debugging
    save_idx = 0 

    model.eval()

    # TODO: convert the lists into a dict to improve readibility
    pq_list, sq_list, rq_list, all_pq_list, all_sq_list, all_rq_list, iou_list, all_iou_list = [], [], [], [], [], [], [], []
    load_time_list, feature_gen_time_list = [], []
    labelling_time_list = {'normalization':[], 'clustering':[], 'label_propogation':[]}
    
    for data in loader:
        torch.cuda.synchronize()
        start_time_load_feature_gen = time.time() 

        data = data.to(device)

        torch.cuda.synchronize()
        end_time_load = time.time()
        print(f'dataloading took {end_time_load-start_time_load_feature_gen} seconds')
        load_time_list.append(end_time_load-start_time_load_feature_gen)

        sem_preds, inst_outs, seed_idx = model(data) # sem_preds size: [NpXNc], inst_outs size: [NpXNe]
        
        torch.cuda.synchronize()
        end_time_load_feature_gen = time.time()
        print(f'feature generation took {end_time_load_feature_gen-end_time_load} seconds')
        feature_gen_time_list.append(end_time_load_feature_gen-end_time_load)

        # break down for each batch
        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for sem_pred, inst_out, sem_label, inst_label, pos in zip(sem_preds.split(sizes), inst_outs.split(sizes), data.y.split(sizes), data.z.split(sizes), data.pos.split(sizes)):
            
            # take argmax of sem_pred and ignore the unlabeled category 0. sem_pred size: [NpX20] -> [1xNp]
            sem_pred[:, ignore_label] = torch.tensor([-float('inf')])
            sem_pred = sem_pred.argmax(-1)

            inst_out_buffer = inst_out

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
            labelling_time_list['normalization'].append(end_time-start_time)

            # debugging test for using seeds from SA module and nearest on pos
            # masked_pos = pos[seed_idx]
            masked_inst_out = inst_out[seed_idx]

            meanshift = MeanShiftEuc(bandwidth=0.6)
            torch.cuda.synchronize()
            start_time = time.time() 

            masked_clustering = meanshift.fit(masked_inst_out)

            torch.cuda.synchronize()
            end_time = time.time()
            print(f'meanshift took {end_time-start_time} seconds')
            labelling_time_list['clustering'].append(end_time-start_time)

            masked_cluster_centers, masked_cluster_labels = masked_clustering.cluster_centers_, masked_clustering.labels_
            maksed_cluster_centers = torch.from_numpy(masked_cluster_centers).float()
            num_inst_ids = len(np.unique(masked_clustering.labels_))
            print(f'number of instance ids: {num_inst_ids}')

            torch.cuda.synchronize()
            start_time = time.time() 
            # clustering = nearest(pos, masked_pos.to(device))
            # clustering = nearest(inst_out, masked_inst_out.to(device))
            clustering = nearest(inst_out, maksed_cluster_centers.to(device))
            # breakpoint()
            # clustering = nearest(pos, pos[global_idx])
            torch.cuda.synchronize()
            end_time = time.time()
            print(f'find nearest neighbors took {end_time-start_time} seconds')
            labelling_time_list['label_propogation'].append(end_time-start_time)

            inst_pred = clustering

            # PQ evaluation
            sem_pred, inst_pred, sem_label, inst_label = preprocess_pred_inst(sem_pred, inst_pred, sem_label, inst_label) # convert to numpy unit32

            absence_label = np.setdiff1d(np.arange(test_dataset.get_n_classes()), sem_label)
            print('absence_label: ', absence_label)
            ignore_absence_label = np.unique(np.append(absence_label, ignore_label))

            evaluator = PanopticEval(test_dataset.get_n_classes(), ignore=ignore_absence_label)
            # evaluator.addBatch(sem_pred, inst_pred, sem_label, inst_label)
            # # Yang: for debugging purpose
            # evaluator.addBatch(sem_label, inst_label, sem_label, inst_label)
            pq, sq, rq, all_pq, all_sq, all_rq = evaluator.getPQ()
            iou, all_iou = evaluator.getSemIoU() # IoU over all classes, IoU for each class

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
        # times
    load_time_list_avr = np.mean(load_time_list, axis=0)
    feature_gen_time_list_avr = np.mean(feature_gen_time_list, axis=0)
    labelling_time_list_avr = {k: np.mean(v, axis=0) for k, v in labelling_time_list.items()}


    return pq_list_avr, sq_list_avr, rq_list_avr, all_pq_list_avr, all_sq_list_avr, all_rq_list_avr, iou_list_avr, all_iou_list_avr, load_time_list_avr, feature_gen_time_list_avr, labelling_time_list_avr


for epoch in range(1): # original is (1, 31)
    # debugging test by loading model
    # state_dict = torch.load('./run_inst/5/Epoch_10_20230218_120949.pth')['net']
    state_dict = torch.load('./run_sem/6/Epoch_33_20230405_133003_semantic_instance.pth')
    model.load_state_dict(state_dict['net'])
    epoch = state_dict['epoch']
    
    pq_list, sq_list, rq_list, all_pq_list, all_sq_list, all_rq_list, iou_list, all_iou_list, load_time_list, feature_gen_time_list, labelling_time_list = test(test_loader, model, if_save_pred=True)
    
    print(pq_list, sq_list, rq_list, all_pq_list, all_sq_list, all_rq_list, iou_list, all_iou_list, load_time_list, feature_gen_time_list, labelling_time_list)

    print(f'Finished testing {epoch}')

    # tensorboarding
    writer.add_scalar('test/pq', pq_list, epoch)
    writer.add_scalar('test/sq', sq_list, epoch)
    writer.add_scalar('test/rq', rq_list, epoch)
    writer.add_scalar('test/mIoU', iou_list, epoch) # miou

    writer.add_scalar('time/dataloading', load_time_list, epoch)
    writer.add_scalar('time/feature generation', feature_gen_time_list, epoch)
    writer.add_scalar('time/labelling/normalization', labelling_time_list['normalization'], epoch)
    writer.add_scalar('time/labelling/clustering', labelling_time_list['clustering'], epoch)
    writer.add_scalar('time/labelling/label_propogation', labelling_time_list['label_propogation'], epoch)

    tboard.add_list(writer, all_pq_list, epoch, DATA_path, 'PQ_all') 
    tboard.add_list(writer, all_sq_list, epoch, DATA_path, 'SQ_all')
    tboard.add_list(writer, all_rq_list, epoch, DATA_path, 'RQ_all')
    tboard.add_list(writer, all_iou_list, epoch, DATA_path, 'IoU_all')
    print(f'Finished tensorboarding {epoch}')

print('All finished!')
