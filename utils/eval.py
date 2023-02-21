import torch
import numpy as np
from utils.clustering import get_inst_masks


def preprocess_pred_inst(sem_pred, inst_pred, sem_label, inst_label):
    """Convert inputs into the right format to use metric evaluation scripts."""
    if isinstance(sem_pred, torch.Tensor):
        sem_pred = sem_pred.cpu().numpy()
    if isinstance(inst_pred, torch.Tensor):
        inst_pred = inst_pred.cpu().numpy()
    if isinstance(sem_label, torch.Tensor):
        sem_label = sem_label.cpu().numpy()
    if isinstance(inst_label, torch.Tensor):
        inst_label = inst_label.cpu().numpy()

    return sem_pred.astype(np.int64), inst_pred.astype(np.int64), sem_label.astype(np.int64), inst_label.astype(np.int64)


def eval_average_precision_inst(inst_pred, sem_pred, sem_label, stuff_list):
    """get thing mask"""
    # add the class 0 to stuff class, which is the unlabeled class
    stuff_unlabeled_list = stuff_list.append(0) 

    # create the mask of semantic predictions where stuff and unlabeled classes appear, and invert it using ~
    thing_mask = ~torch.isin(sem_pred, stuff_unlabeled_list)

    # get the instance/semantic predictions where only thing classes exist, and length is reduced
    thing_inst_pred = inst_pred[thing_mask] # do contain 0 values
    thing_sem_pred = sem_pred[thing_mask] # do not contain 0 values

    """get iou"""
    # get unique inst_ids and corresponding frequencies
    thing_inst_unique_values, frequencies = torch.unique(thing_inst_pred, return_counts=True)
    num_clusters = len(thing_inst_unique_values)

    # get inst_id mask
    thing_inst_masks = get_inst_masks(num_clusters, thing_inst_pred) # 0-1 float mask, cannot reduce length

    # get inst_id-masked semantic prediction
    id_masked_thing_sem_pred = thing_inst_masks * thing_sem_pred # dim(id_masked_thing_sem_pred) = [Number of clusters X Number of points in interest]

    # majority vote of semantic labels in a cluster and propogate through all points in the cluster
    majority_vote_sem_pred = id_masked_thing_sem_pred[id_masked_thing_sem_pred != 0].view(num_clusters, -1).mode(dim=-1) # [1 X Number of clusters], mode for each row
    thing_inst_sem_pred = thing_inst_masks * majority_vote_sem_pred.view(len(majority_vote_sem_pred), 1) # [Number of clusters X Number of points in interest]

    # get iou between thing_inst_sem_pred and thing_inst_sem_gt
    thing_inst_sem_pred = torch.sum(thing_inst_sem_pred, 0) # sum over column, as only one value exists in each column
    thing_inst_sem_label = sem_label[thing_mask]

    """get confusion matrix (tp/fp/fn)"""

    """get recall/precision"""

    """get average precision"""
    # put 1 here just to avoid bugs
    average_precision = 1

    return average_precision

