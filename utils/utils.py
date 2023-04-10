import torch
import numpy as np
import yaml
import os

def calc_iou_per_cat(pred, target, num_classes, ignore_index):
    """The iou for ignored category will always be zero
    """
    # Initialize a tensor to store the IoU for each class
    iou_per_class = torch.zeros(num_classes)

    pred[:, ignore_index] = torch.tensor([-float('inf')])
    pred = pred.argmax(-1)
    # Iterate over each class
    for c in range(num_classes):
        if c == ignore_index:
            continue
        else:
            # Create masks that identify the predicted and target elements for class c
            pred_mask = (pred == c)
            target_mask = (target == c)

            # Compute the area of the intersection of the predicted and target tensors for class c
            intersection = (pred_mask * target_mask).sum()

            # Compute the area of the union of the predicted and target tensors for class c
            union = pred_mask.sum() + target_mask.sum() - intersection

            # Compute the IoU for class c as the ratio of the intersection to the union
            iou_per_class[c] = intersection / union

    # Return the IoU tensor
    return iou_per_class

def calc_miou(ious, num_classes, ignore_label):
    """As the iou for ignored category is zero, the sum will not change but number of needed categoy is minused by one
    """
    if ignore_label == False:
        num_classes = num_classes
    else:
        num_classes -= 1
    mask = torch.isnan(ious)
    # print(mask.nonzero())
    tnc = torch.tensor(num_classes) - torch.sum(mask)
    ious = ious.masked_fill(mask,0)
    siou = torch.sum(ious)
    # print(tnc,siou)
    return siou/tnc

def averaging_ious(ious):
    """This function is used to calculate the average iou for each category by processing the nan value. 
    The nan value in ious means that the category is not present in the point cloud data so when calcualting average iou this nan value should be
    removed. 
    """
    # Create a tensor to store the number of NaN values for each category (column)
    nan_count = torch.isnan(ious).sum(dim=0, dtype=torch.float)

    # Replace NaN values with 0
    ious = ious.masked_fill(torch.isnan(ious), 0)

    # Take the sum of each column
    row_sums = ious.sum(dim=0)


    # Divide the sum along each category by the number of rows minus the number of NaN values
    processed_iou = row_sums / (ious.shape[0] - nan_count)

    return processed_iou

# def inst_eval(inst_label, inst_pred, sem_label, sem_pred):
#     '''Evaluation part, already use predictions to mask out stuff and unlabeled classes'''
#     # get thing mask
#     thing_mask = torch.where(inst_label == 0)
    
#     # TODO: be careful of the situation where the entire inst_label list is masked out, return is tensor[[]]
#     thing_masked_inst_label = inst_label[thing_mask] # get a reduced tensor list [1XNtp], where Ntp is number of elements unequal to 0
#     thing_masked_inst_pred = inst_pred[thing_mask]


def get_xentropy_class_string(label, DATA_path):
    DATA = yaml.safe_load(open(DATA_path, 'r'))
    learning_map_inv, labels= DATA['learning_map_inv'], DATA['labels']
    # print(labels[learning_map_inv[label]])
    return labels[learning_map_inv[label]]


def save_tensor_to_disk(label, filename, filepath, run, sequence):

    filepath = os.path.join(filepath, run, sequence, filename)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(filepath, 'wb') as f:
        torch.save(label, f)


def open_tensor_from_disk(label, filepath):
    with opern(filepath, 'rb') as f:
        things_to_load = torch.load(f)
    return things_to_load

def save_pred_to_disk(pred, sequence, save_pred_path, run, save_idx):
    save_path = os.path.join(save_pred_path, run, 'sequences', sequence, 'predictions') # e.g. panoptic_data/0/sequences/08/predictions
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = f'{save_idx:06d}.label'

    # create file and write binary data
    with open(os.path.join(save_path, filename), 'wb') as f:
        # f.write(np.float32(pred)) # save as float 32 instead of float 64
        f.write(pred)

def set_random_seeds(seed):
    # Set random seeds for NumPy
    np.random.seed(seed)
    random.seed(seed)

    # Set random seeds for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False