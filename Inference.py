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
from model.Randlanet_mlp import RandlaNet_mlp

# define the gpu index to train
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# data_path = '/Volumes/scratchdata/kitti/dataset/'
# data_path = '/Volumes/scratchdata_smb/kitti/dataset/' # alternative path
data_path = '/Volumes/mrgdatastore6/ThirdPartyData/semantic_kitti/dataset'
# DATA_path = '/home/yanghou/project/Panoptic-Segmentation/semantic-kitti.yaml' #
DATA_path = './semantic-kitti.yaml' # for running in docker
save_path = './run_inst_loss'
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
run = str(52)
save_path = os.path.join(save_path, run) # model state path
if not os.path.exists(save_path):
    os.makedirs(save_path)
print('Run Number is :', run)



# create a SummaryWriter to write logs to a log directory
log_path = 'log_inst_loss'
log_path = os.path.join(log_path, run) # train/test info path
writer = SummaryWriter(log_dir=log_path, filename_suffix=time.strftime("%Y%m%d_%H%M%S"))

# device = torch.device('cpu') # only for debugging
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net(test_dataset.get_n_classes()).to(device)
# model = RandlaNet(num_features=3,
#         num_classes=20,
#         decimation=4,
#         num_neighbors=16).to(device)
asis = {'feature_dim': 32, 'inst_emb_dim': 5, 'num_class': 20, 'k':30, 'distance_threshold': 0.5, 'norm_degree': 2}
model = RandlaNet_mlp(num_features=3,
        num_classes=20,
        decimation=4,
        num_neighbors=16, 
        dim_inst_out=5, 
        use_asis=False,
        asis=asis).to(device)

@torch.no_grad()
def test_pos(loader, model, bandwidth, if_save_pred, normalization, seeding_clustering):
                                #  ['car', 'bicycle','motorcycle', 'truck', 'other-vehicle','person', 'bicyclist', 'motorcyclist']
    thing_in_xentropy = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
                            #  ['road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
    stuff_in_xentropy = torch.tensor([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    # ic = 0 # for quick debugging
    save_idx = 0 
    count = 0
    model.eval()


    # TODO: convert the lists into a dict to improve readibility
    pq_list, sq_list, rq_list, all_pq_list, all_sq_list, all_rq_list, iou_list, all_iou_list = [], [], [], [], [], [], [], []
    load_time_list, feature_gen_time_list = [], []
    labelling_time_list = {'normalization':[], 'clustering':[], 'label_propogation':[]}
    
    print(f'Start clustering useing positional coordinates')

    for data in loader:

        # if count >= 3:
        #     break
        #     count += 1
        #     continue

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
            empty_bool = False
            # # overide instance feature to be positional coordinates
            # inst_out = pos
            

            # take argmax of sem_pred and ignore the unlabeled category 0. sem_pred size: [NpX20] -> [1xNp]
            sem_pred[:, ignore_label] = torch.tensor([-float('inf')])
            sem_pred_unmasked = sem_pred.argmax(-1) # [1XNp]
            unlabel_mask = torch.isin(sem_label, torch.tensor(ignore_label).to(device)) 
            sem_pred = sem_pred_unmasked.masked_fill(unlabel_mask, 0) # exclude value that is in ignore_label
            # print('sem_pred before unlabled masking:', sem_pred_unmasked)
            # print('sem_pred after unlabled masking:', sem_pred)
            # print('sem_label:', data.y)
            # print('inst_label:', data.z)
            print('number of ground truth instance ids including 0:', torch.unique(data.z))

            # get thing_mask from sem_pred
            thing_mask = torch.isin(sem_pred, thing_in_xentropy.to(device))
            # masked_inst_out = inst_out.masked_fill(thing_mask.repeat(inst_out.shape[0]), 0) # keep value that is in thing_mask
            if seeding_clustering == True:
                # masked_inst_out = inst_out[seed_idx][thing_mask[seed_idx]]
                # breakpoint()
                masked_inst_out = inst_out[seed_idx]
                # print('masked_inst_out for seeding clustering', masked_inst_out)
            else:    
                masked_inst_out = inst_out[thing_mask]
                # check if masked_inst_out is empty
                if masked_inst_out.numel() == 0:
                    print('if masked_inst_out.numel() == 0:')
                    masked_inst_out = inst_out
                    empty_bool = True
                print('masked_inst_out for direct clustering', masked_inst_out)

            if normalization == True:
                torch.cuda.synchronize()
                start_time = time.time()

                # if there is only one point feature, skip normalization
                if masked_inst_out.shape[0] == 1:
                    continue
                else:
                    # normalize masked instance features by mean and standard deviation for better clustering
                    masked_inst_out_mean = torch.mean(masked_inst_out, dim=0)
                    # print('inst_out_mean', inst_out_mean)
                    masked_inst_out_std = torch.std(masked_inst_out, dim=0)
                    # print('inst_out_std', inst_out_std)
                    masked_inst_out = (masked_inst_out - masked_inst_out_mean) / masked_inst_out_std
                    # print(' masked_inst_out after normalization',  masked_inst_out)
                    # print('normalization mean', masked_inst_out_mean)
                    # print('normalization std', masked_inst_out_std)
                
                torch.cuda.synchronize()
                end_time = time.time()
                print(f'normalization took {end_time-start_time} seconds')
                labelling_time_list['normalization'].append(end_time-start_time)
                # distances = torch.cdist(masked_inst_out, masked_inst_out)
                # estimated_bandwidth = torch.median(distances.view(-1)).item()
                # print('estimated_bandwidth', estimated_bandwidth)

                print('normalized masked inst out', masked_inst_out)

            meanshift = MeanShiftEuc(bandwidth=bandwidth)
            torch.cuda.synchronize()
            start_time = time.time()

            masked_clustering = meanshift.fit(masked_inst_out)
            
            # breakpoint()

            torch.cuda.synchronize()
            end_time = time.time()
            print(f'meanshift took {end_time-start_time} seconds')
            labelling_time_list['clustering'].append(end_time-start_time)

            masked_cluster_centers, masked_cluster_labels = masked_clustering.cluster_centers_, masked_clustering.labels_

            num_inst_ids = len(np.unique(masked_clustering.labels_))
            print(f'number of prediction instance ids including 0: {num_inst_ids}')
            
            # TODO: move all the computation either in cpu or gpu to avoid copying between the devices
            if seeding_clustering == True:
                inst_pred = torch.zeros_like(thing_mask, dtype=torch.long)
                
                torch.cuda.synchronize()
                start_time = time.time()

                # check if it is empty
                if inst_out[thing_mask].numel() == 0:
                    # print('if inst_out[thing_mask].numel() == 0:')
                    empty_bool = True
                    clustering = nearest(inst_out, torch.tensor(masked_cluster_centers, dtype=torch.float).to(device))
                else:
                    clustering = nearest(inst_out[thing_mask], torch.tensor(masked_cluster_centers, dtype=torch.float).to(device)) + 1
                # inst_pred[seed_idx][thing_mask[seed_idx]] = clustering
                # breakpoint()

                torch.cuda.synchronize()
                end_time = time.time()
                print(f'find nearest neighbors took {end_time-start_time} seconds')
                labelling_time_list['label_propogation'].append(end_time-start_time)

                if empty_bool == True:
                    # print('Line215 if empty_bool == True:')
                    inst_pred = clustering
                else:
                    inst_pred[thing_mask] = clustering
                # print('inst_pred after unmasking:', inst_pred, inst_pred.shape)
                # print(f'number of final prediction instance ids including 0: {torch.unique(inst_pred)}')
                
                # # to visualize the seeding points
                # inst_pred = torch.zeros(len(inst_label))
                # inst_pred[seed_idx][thing_mask[seed_idx]] = 1
                
            else:
                if empty_bool == False:
                    unmasked_inst_pred = masked_cluster_labels
                    print('inst_pred before unmasking:', unmasked_inst_pred, unmasked_inst_pred.shape, unmasked_inst_pred.dtype)
                    inst_pred = torch.zeros_like(thing_mask, dtype=torch.long)
                    inst_pred[thing_mask] =  torch.from_numpy(unmasked_inst_pred).to(device)
                    print('inst_pred after unmasking:', inst_pred, inst_pred.shape)
                    print(f'number of final prediction instance ids including 0: {torch.unique(inst_pred)}')
                else:
                    inst_pred = torch.zeros_like(thing_mask, dtype=torch.long)
            # breakpoint()

            # breakpoint()

            # PQ evaluation
            sem_pred, inst_pred, sem_label, inst_label = preprocess_pred_inst(sem_pred, inst_pred, sem_label, inst_label) # convert to numpy unit32

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
            mask = np.isin(np.arange(test_dataset.get_n_classes()), absence_label)
            all_pq[mask] = np.nan
            all_sq[mask] = np.nan
            all_rq[mask] = np.nan
            all_iou[mask] = np.nan
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
    all_pq_list_avr = np.nanmean(all_pq_list, axis=0)
    all_sq_list_avr = np.nanmean(all_sq_list, axis=0)
    all_rq_list_avr = np.nanmean(all_rq_list, axis=0)
    iou_list_avr = np.mean(iou_list, axis=0)
    all_iou_list_avr = np.nanmean(all_iou_list, axis=0)
    
        # times
    load_time_list_avr = np.mean(load_time_list, axis=0)
    feature_gen_time_list_avr = np.mean(feature_gen_time_list, axis=0)
    labelling_time_list_avr = {k: np.mean(v, axis=0) for k, v in labelling_time_list.items()}
    # breakpoint()
    return pq_list_avr, sq_list_avr, rq_list_avr, all_pq_list_avr, all_sq_list_avr, all_rq_list_avr, iou_list_avr, all_iou_list_avr, load_time_list_avr, feature_gen_time_list_avr, labelling_time_list_avr


@torch.no_grad()
def test_feature(loader, model, bandwidth, if_save_pred, normalization, seeding_clustering):
                                #  ['car', 'bicycle','motorcycle', 'truck', 'other-vehicle','person', 'bicyclist', 'motorcyclist']
    thing_in_xentropy = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
                            #  ['road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
    stuff_in_xentropy = torch.tensor([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    # ic = 0 # for quick debugging
    save_idx = 0 
    count = 0
    model.eval()


    # TODO: convert the lists into a dict to improve readibility
    pq_list, sq_list, rq_list, all_pq_list, all_sq_list, all_rq_list, iou_list, all_iou_list = [], [], [], [], [], [], [], []
    load_time_list, feature_gen_time_list = [], []
    labelling_time_list = {'normalization':[], 'clustering':[], 'label_propogation':[]}

    for data in loader:

        # if count >= 3:
        #     break
        #     count += 1
        #     continue

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
            empty_bool = False

            # take argmax of sem_pred and ignore the unlabeled category 0. sem_pred size: [NpX20] -> [1xNp]
            sem_pred[:, ignore_label] = torch.tensor([-float('inf')])
            sem_pred_unmasked = sem_pred.argmax(-1) # [1XNp]
            unlabel_mask = torch.isin(sem_label, torch.tensor(ignore_label).to(device)) 
            sem_pred = sem_pred_unmasked.masked_fill(unlabel_mask, 0) # exclude value that is in ignore_label
            # print('sem_pred before unlabled masking:', sem_pred_unmasked)
            # print('sem_pred after unlabled masking:', sem_pred)
            # print('sem_label:', data.y)
            # print('inst_label:', data.z)
            print('number of ground truth instance ids including 0:', torch.unique(data.z))

            # get thing_mask from sem_pred
            thing_mask = torch.isin(sem_pred, thing_in_xentropy.to(device))
            # masked_inst_out = inst_out.masked_fill(thing_mask.repeat(inst_out.shape[0]), 0) # keep value that is in thing_mask
            if seeding_clustering == True:
                # masked_inst_out = inst_out[seed_idx][thing_mask[seed_idx]]
                # breakpoint()
                masked_inst_out = inst_out[seed_idx]
                # print('masked_inst_out for seeding clustering', masked_inst_out)
            else:    
                masked_inst_out = inst_out[thing_mask]
                # check if masked_inst_out is empty
                if masked_inst_out.numel() == 0:
                    print('if masked_inst_out.numel() == 0:')
                    masked_inst_out = inst_out
                    empty_bool = True
                print('masked_inst_out for direct clustering', masked_inst_out)

            if normalization == True:
                torch.cuda.synchronize()
                start_time = time.time()

                # if there is only one point feature, skip normalization
                if masked_inst_out.shape[0] == 1:
                    continue
                else:
                    # normalize masked instance features by mean and standard deviation for better clustering
                    masked_inst_out_mean = torch.mean(masked_inst_out, dim=0)
                    # print('inst_out_mean', inst_out_mean)
                    masked_inst_out_std = torch.std(masked_inst_out, dim=0)
                    # print('inst_out_std', inst_out_std)
                    masked_inst_out = (masked_inst_out - masked_inst_out_mean) / masked_inst_out_std
                    # print(' masked_inst_out after normalization',  masked_inst_out)
                    # print('normalization mean', masked_inst_out_mean)
                    # print('normalization std', masked_inst_out_std)
                
                torch.cuda.synchronize()
                end_time = time.time()
                print(f'normalization took {end_time-start_time} seconds')
                labelling_time_list['normalization'].append(end_time-start_time)
                # distances = torch.cdist(masked_inst_out, masked_inst_out)
                # estimated_bandwidth = torch.median(distances.view(-1)).item()
                # print('estimated_bandwidth', estimated_bandwidth)

                print('normalized masked inst out', masked_inst_out)

            meanshift = MeanShiftEuc(bandwidth=bandwidth)
            torch.cuda.synchronize()
            start_time = time.time()

            masked_clustering = meanshift.fit(masked_inst_out)
            
            # breakpoint()

            torch.cuda.synchronize()
            end_time = time.time()
            print(f'meanshift took {end_time-start_time} seconds')
            labelling_time_list['clustering'].append(end_time-start_time)

            masked_cluster_centers, masked_cluster_labels = masked_clustering.cluster_centers_, masked_clustering.labels_

            num_inst_ids = len(np.unique(masked_clustering.labels_))
            print(f'number of prediction instance ids including 0: {num_inst_ids}')
            
            # TODO: move all the computation either in cpu or gpu to avoid copying between the devices
            if seeding_clustering == True:
                inst_pred = torch.zeros_like(thing_mask, dtype=torch.long)
                
                torch.cuda.synchronize()
                start_time = time.time()

                # check if it is empty
                if inst_out[thing_mask].numel() == 0:
                    # print('if inst_out[thing_mask].numel() == 0:')
                    empty_bool = True
                    clustering = nearest(inst_out, torch.tensor(masked_cluster_centers, dtype=torch.float).to(device))
                else:
                    clustering = nearest(inst_out[thing_mask], torch.tensor(masked_cluster_centers, dtype=torch.float).to(device)) + 1
                # inst_pred[seed_idx][thing_mask[seed_idx]] = clustering
                # breakpoint()

                torch.cuda.synchronize()
                end_time = time.time()
                print(f'find nearest neighbors took {end_time-start_time} seconds')
                labelling_time_list['label_propogation'].append(end_time-start_time)

                if empty_bool == True:
                    # print('Line215 if empty_bool == True:')
                    inst_pred = clustering
                else:
                    inst_pred[thing_mask] = clustering
                # print('inst_pred after unmasking:', inst_pred, inst_pred.shape)
                # print(f'number of final prediction instance ids including 0: {torch.unique(inst_pred)}')
                
                # # to visualize the seeding points
                # inst_pred = torch.zeros(len(inst_label))
                # inst_pred[seed_idx][thing_mask[seed_idx]] = 1
                
            else:
                if empty_bool == False:
                    unmasked_inst_pred = masked_cluster_labels
                    print('inst_pred before unmasking:', unmasked_inst_pred, unmasked_inst_pred.shape, unmasked_inst_pred.dtype)
                    inst_pred = torch.zeros_like(thing_mask, dtype=torch.long)
                    inst_pred[thing_mask] =  torch.from_numpy(unmasked_inst_pred).to(device)
                    print('inst_pred after unmasking:', inst_pred, inst_pred.shape)
                    print(f'number of final prediction instance ids including 0: {torch.unique(inst_pred)}')
                else:
                    inst_pred = torch.zeros_like(thing_mask, dtype=torch.long)
            # breakpoint()

            # breakpoint()

            # PQ evaluation
            sem_pred, inst_pred, sem_label, inst_label = preprocess_pred_inst(sem_pred, inst_pred, sem_label, inst_label) # convert to numpy unit32

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
            mask = np.isin(np.arange(test_dataset.get_n_classes()), absence_label)
            all_pq[mask] = np.nan
            all_sq[mask] = np.nan
            all_rq[mask] = np.nan
            all_iou[mask] = np.nan
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
    all_pq_list_avr = np.nanmean(all_pq_list, axis=0)
    all_sq_list_avr = np.nanmean(all_sq_list, axis=0)
    all_rq_list_avr = np.nanmean(all_rq_list, axis=0)
    iou_list_avr = np.mean(iou_list, axis=0)
    all_iou_list_avr = np.nanmean(all_iou_list, axis=0)
    
        # times
    load_time_list_avr = np.mean(load_time_list, axis=0)
    feature_gen_time_list_avr = np.mean(feature_gen_time_list, axis=0)
    labelling_time_list_avr = {k: np.mean(v, axis=0) for k, v in labelling_time_list.items()}
    # breakpoint()
    return pq_list_avr, sq_list_avr, rq_list_avr, all_pq_list_avr, all_sq_list_avr, all_rq_list_avr, iou_list_avr, all_iou_list_avr, load_time_list_avr, feature_gen_time_list_avr, labelling_time_list_avr


for epoch in range(1): # original is (1, 31)
    # debugging test by loading model
    state_dict = torch.load('./run_inst_loss/48/Epoch_69_20230427_032341_semantic_instance.pth')
    model.load_state_dict(state_dict['net'])
    epoch = state_dict['epoch']
    
    # pq_list, sq_list, rq_list, all_pq_list, all_sq_list, all_rq_list, iou_list, all_iou_list, load_time_list, feature_gen_time_list, labelling_time_list = test_feature(test_loader, model, bandwidth=0.1, if_save_pred=True, normalization=False, seeding_clustering=True)
    pq_list, sq_list, rq_list, all_pq_list, all_sq_list, all_rq_list, iou_list, all_iou_list, load_time_list, feature_gen_time_list, labelling_time_list = test_feature(test_loader, model, bandwidth=0.2, if_save_pred=True, normalization=True, seeding_clustering=False)
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