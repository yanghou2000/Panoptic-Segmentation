# This dataset is replaced with torch.utils.data instead of torch_geometric.dataset
# return points, label

import os
import numpy as np
from numpy.random import permutation
import torch
# from torch_geometric.data import Dataset, Data
from torch.utils.data import Dataset
import yaml
# remember to bring back pc
# from PointNet2_sem_seg_msg.PointNet2_sem_seg_msg_utils import pc_normalize 
import torch.nn.functional as F
# from PointNet2_sem_seg_msg_utils import pc_normalize # for debugging in loader file

# for finding the minimum lengh of sequence
from torch.utils.data import DataLoader

import time


class SemanticKitti(Dataset):
    EXTENSIONS_SCAN = ['.bin']
    EXTENSIONS_LABEL = ['.label']
    
    np.random.seed(42)

    def __init__(self, dataset_dir, sequences, DATA_dir):
        # super().__init__(root=dataset_dir, transform=None, pre_transform=None, pre_filter=None)
        DATA = yaml.safe_load(open(DATA_dir, 'r'))
        self.learning_map, self.learning_map_inv, self.labels, self.content = DATA['learning_map'], DATA['learning_map_inv'], DATA['labels'], DATA['content']
        self.sequences = sequences
        self.nclasses = len(self.learning_map_inv)
        self.scan_names, self.label_names = [], []

        # Iterate through sequences 
        for sequence in self.sequences:

            print(f'Using Sequence Number: {sequence}')

            #does scan/label folder exist
            self.scan_paths = os.path.join(dataset_dir, "sequences", sequence, "velodyne")
            self.label_paths = os.path.join(dataset_dir, "sequences", sequence, "labels")                    
            
            if os.path.isdir(self.scan_paths):
                print("Sequence folder exists! Using sequence from %s" % self.scan_paths)
            else:
                print("Sequence folder doesn't exist! Exiting...")
                quit()

            if os.path.isdir(self.label_paths):
                print("Labels folder exists! Using labels from %s" % self.label_paths)
            else:
                print("Labels folder doesn't exist! Exiting...")
                quit()

            # populate the pointclouds/labels
            scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(self.scan_paths)) for f in fn if not f.startswith('.')]
            label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(self.label_paths)) for f in fn if not f.startswith('.')]
            
            # extend list
            self.scan_names.extend(scan_names)
            self.label_names.extend(label_names)
            
            print(f'Number of scan/label files:{len(self.label_names)} in {sequence}')

        # sort for correspondance
        self.scan_names.sort()
        self.label_names.sort()

        # check that there are same amount of labels and scans
        assert(len(self.label_names) == len(self.scan_names))

    def __len__(self):
        return len(self.label_names)

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]

    def __getitem__(self, idx):
        """ Open raw scan and fill in attributes
        """
        scan_name = self.scan_names[idx]

        # check filename is string
        if not isinstance(scan_name, str):
            raise TypeError("Filename should be string type, "
                        "but was {type}".format(type=str(type(scan_name))))

        # check extension is a laserscan
        if not any(scan_name.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(scan_name, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]    # get xyz

        # # normalization
        # points = pc_normalize(points)

        # number of points before sampling
        num_points_bs = points.shape[0]
        # print(f'Number of points in scan: {num_points}')
        # print(f'Number of classes: {self.nclasses}')

        """ Open orignal label and fill in attributes
        """
        label_name = self.label_names[idx]
        # check filename is string
        if not isinstance(label_name, str):
            raise TypeError("Filename should be string type, "
                        "but was {type}".format(type=str(type(label_name))))

        # check extension is a laserscan
        if not any(label_name.endswith(ext) for ext in self.EXTENSIONS_LABEL):
            raise RuntimeError("Filename extension is not valid label file.")

        # if all goes well, open label
        label = np.fromfile(label_name, dtype=np.int32)
        label = label.reshape((-1))

        # sem_label and isntance label
        sem_label = label & 0xFFFF
        inst_label = label >> 16

        # label_string = [self.get_original_class_string(i) for i in sem_label] # suppress for speed
        map_label = [self.to_xentropy(i) for i in sem_label]
        # map_label_string = [self.get_xentropy_class_string(i) for i in map_label] # suppress for speed      
        map_label = np.array(map_label) # convert to numpy array for indexing

        # downsample points and labels to desired sizes if required, else add padding
        if num_points_bs >= 120000:
            sample_indices = permutation(120000)
            points = points[sample_indices]
            map_label = map_label[sample_indices]
        else:
            # TODO: tidy-up points/map_labels data structure for consistency
            points = torch.from_numpy(points)
            residual_num = 120000 - num_points_bs # number of padding needed
            points = F.pad(points, (0, 0, 0, residual_num), "constant", 0)
            map_label = np.concatenate((map_label, np.zeros(residual_num)), axis=None)


        
        num_points_as = points.shape[0] # number of points after sampling

        # convert from numpy to tensor if not tensor, default to float64
        if not torch.is_tensor(points):
            points = torch.from_numpy(points)
        
        # # convert y to one_hot_label
        # onehot_label = self.to_onehot(map_label,self.nclasses, num_points)
        # onehot_label = torch.from_numpy(onehot_label)

        # pos:Node position matrix with shape [num_nodes, num_dimensions]
        # data = Data(pos=points, y=label.long)
        map_label = torch.from_numpy(np.array(map_label))
        # data = Data(pos=points, y=map_label.long())

        # to fit new model, input points should be [C, N] instead of [N, C]
        # map_label.size: [1, N]
        return points.permute(1, 0), map_label.long()
    
    def get_n_classes(self):
        return self.nclasses

    def get_original_class_string(self, idx):
        return self.labels[idx]

    def get_xentropy_class_string(self, idx):
        print(self.labels[self.learning_map_inv[idx]])
        return self.labels[self.learning_map_inv[idx]]

    def to_original(self, label):
        # put label in original values
        return self.map(label, self.learning_map_inv)

    def to_xentropy(self, label):
        # put label in xentropy values
        return self.map(label, self.learning_map)

    def to_onehot(self, label, nclasses, num_points):
        # initialize a matrix with dimension[nclasses, num_points]
        self.onehot_label = np.zeros((num_points, nclasses))
        for i, j in enumerate(label):
            self.onehot_label[i][j] = 1
        return self.onehot_label

    def map_loss_weight(self):
        epsilon_w = 0.001
        content = torch.zeros(self.nclasses)
        for cl, freq in self.content.items():
            x_cl = self.to_xentropy(cl)
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)
        return self.loss_w


#Define a main function
if __name__=='__main__':
    #get learning_map and learning_map_inv
    DATA_dir = './semantic-kitti.yaml'
    # mydataset = SemanticKitti(dataset_dir='/Volumes/scratchdata/kitti/dataset/', 
    #                                 sequences=['00'], 
    #                                 DATA_dir = DATA_dir)
    # print(mydataset[0])
    # x, y = mydataset[0]
    # print(len(y))
    # print(mydataset[0].y[6:10,:])
    # print(f'pos: {type(mydataset[0].pos)}')
    # print(mydataset.map_loss_weight())
    # print(f'len: {mydataset.__len__()}')
    # print(f'original label: {mydataset[0].y[-10:]}, type: {type(mydataset[0].y)}')
    # print(f'orignial label string: {mydataset[0].label_string[-10:]}')
    # print(f'mapped label: {mydataset[0].map_label[-10:]}')
    # print(f'mapped label string: {mydataset[0].map_label_string[-10:]}')
    torch.manual_seed(42)

    debugging_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    train_dataset = SemanticKitti(dataset_dir='/Volumes/scratchdata/kitti/dataset/', 
                                sequences= debugging_sequences, 
                                DATA_dir=DATA_dir)

    

    train_loader_pin_true = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, persistent_workers=torch.cuda.is_available(), pin_memory=False,
                          drop_last=True)
    
    start = time.time()

    # print()

    for i, (points, labels), in enumerate(train_loader_pin_true):
        # leng = points.size(dim=-1)
        # print(leng)
        # print(f'labels.size(): {labels.size()}')
        
        if i == 100:
            finish = time.time()
            print(f'duration_pin_true: {int(finish)-int(start)}')
            break