import math
import operator
import time

import numpy as np 
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

import torch
from torch import exp, sqrt

# This class if copied from repo: https://github.com/thanhkaist/MeanShiftClustering/blob/master/mean-shift-pytorch-gpu.py
class MeanShift_GPU():
    ''' Do meanshift clustering with GPU support'''
    def __init__(self,bandwidth = 2.5, batch_size = 1000, max_iter = 10, eps = 1e-5, check_converge = False):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.bandwidth = bandwidth
        self.eps = eps # use for check converge
        self.cluster_eps = 1e-1 # use for check cluster
        self.check_converge = check_converge # Check converge will take 1.5 time longer
          
    def distance_batch(self,a,B):
        ''' Return distance between each element in a to each element in B'''
        return sqrt(((a[None,:] - B[:,None])**2)).sum(2)
    
    def distance(self,a,b):
        return np.sqrt(((a-b)**2).sum())
    
    def fit(self,data):
        with torch.no_grad():
            n = len(data)
            if not data.is_cuda:
                data_gpu = data.cuda()
            
            X = data_gpu.clone()
            #X = torch.from_numpy(np.copy(data)).cuda()
            
            for _ in range(self.max_iter):
                max_dis = 0;
                for i in range(0,n,self.batch_size):
                    s = slice(i,min(n,i+ self.batch_size))
                    if self.check_converge:
                        dis = self.distance_batch(X,X[s])
                        max_batch = torch.max(dis)
                        if max_dis < max_batch:
                            max_dis = max_batch;
                        weight = dis
                        weight = self.gaussian(dis, self.bandwidth)
                    else:
                        weight = self.gaussian(self.distance_batch(X,X[s]), self.bandwidth)
                    num = (weight[:,:,None]*X).sum(dim=1)
                    X[s] = num / weight.sum(1)[:,None]                    
                    
                #import pdb; pdb.set_trace()
                #Check converge
                if self.check_converge:
                    if max_dis < self.eps:
                        print("Converged")
                        break
            
            end_time = time.time()
            print("algorithm time (s)", end_time- begin_time)
            # Get center and labels
            if True:
                # Convert to numpy cpu show better performance
                points = X.cpu().data.numpy()
                labels, centers = self.cluster_points(points)
            else:
                # use GPU
                labels, centers = self.cluster_points(points)
                
            return labels,centers
        
    def gaussian(self,dist,bandwidth):
        return exp(-0.5*((dist/bandwidth))**2)/(bandwidth*math.sqrt(2*math.pi))
        
    def cluster_points(self, points):
        cluster_ids = []
        cluster_idx = 0
        cluster_centers = []

        for i, point in enumerate(points):
            if(len(cluster_ids) == 0):
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:
                for j,center in enumerate(cluster_centers):
                    dist = self.distance(point, center)
                    if(dist < self.cluster_eps):
                        cluster_ids.append(j)
                if(len(cluster_ids) < i + 1):
                    cluster_ids.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1
        return cluster_ids, cluster_centers



def cluster(prediction, bandwidth, n_jobs):
    '''Inputs and outputs are tensors. Intermediate states are numpy. If estimate_bandwidth=True, use estimated bandwidth.
    '''
    # TODO: prediction needs to be swtiched between cpu and gpu. Use GPU based clustering algorithm instead
    prediction = prediction.cpu()

    # from tensor to np
    prediction = np.array(prediction)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=n_jobs)
    #print ('Mean shift clustering, might take some time ...')
    #tic = time.time()
    ms.fit(prediction)
    #print ('time for clustering', time.time() - tic)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    num_clusters = cluster_centers.shape[0]

    return num_clusters, torch.from_numpy(labels), torch.from_numpy(cluster_centers)

# TODO: compose the functions into a class to evaluate the instance segmentation performance using mAP
def thing_masked_inst_sem_preds(inst_pred, sem_pred, stuff_list):
    """return mask that corresponds to thing classes only, ignoring stuff classes and unlabel class based on
    semantic predictions
    """
    # add the class 0 to stuff class, which is the unlabeled class
    stuff_unlabeled_list = stuff_list.append(0) 

    # create the mask of semantic predictions where stuff and unlabeled classes appear, and invert it using ~
    thing_mask = ~torch.isin(sem_pred, stuff_unlabeled_list)

    # get the instance/semantic predictions where only thing classes exist, and length is reduced
    thing_inst_pred = inst_pred[thing_mask]
    thing_sem_pred = sem_pred[thing_mask]

    return thing_inst_pred, thing_sem_pred


def get_inst_masks(num_clusters, labels):
    '''return a tensor list that contains masks for every instance id
    Inputs: 
        num_clusters: dtype=numpy
        labels: dtype=tensor, dim=[1XNp]
    Outputs:
        mask: dtype=tensor, dim=[num_clustersXNp]
    '''
    Np = len(labels)
    
    instance_masks = torch.zeros(num_clusters, Np)

    for mask_id in range(num_clusters):
        instance_masks[mask_id] = torch.eq(labels, mask_id).long()

    return instance_masks


# def get_inst_to_sem_labels(mask, sem_labels)
"""use majoirty vote to propogate semantic label to see if semantic performance improves
"""

