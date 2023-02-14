import torch
import numpy as np
from utils.clustering import get_inst_masks


# def majority_vote_propogate(inst_pred, sem_pred):
#     """Get the majority vote of semantic predictions for points in clusters of instance predictions, and propogate the majority vote for all points in the cluster so that
#     all points in a cluster have the same semantic label. Prepare the labels to plug in semantic_kitti_eval_np.py.
#     """
def preprocess_pred_inst(sem_pred, inst_pred, sem_label, inst_label):
    sem_pred, inst_pred, sem_label, inst_label = sem_pred.cpu(), inst_pred.cpu(), sem_label.cpu(), inst_label.cpu()
    # reshape(1, -1) if needed
    sem_pred = np.array(sem_pred, dtype=np.int64)
    inst_pred = np.array(inst_pred, dtype=np.int64)
    sem_label = np.array(sem_label, dtype=np.int64)
    inst_label = np.array(inst_label, dtype=np.int64)
    return sem_pred, inst_pred, sem_label, inst_label


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

