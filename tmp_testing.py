import torch
import numpy as np
import math

# mious = []

# a = torch.tensor(0.2882)
# b = torch.tensor(0.3278)

# mious.append(a)
# mious.append(b)
# mious = torch.as_tensor(mious)

# miou_avr = torch.sum(mious) / mious.size()[0]

# print(miou_avr, miou_avr.dtype)


# def nan_process(tensor):
#     # Create a tensor to store the number of NaN values for each category (column)
#     nan_count = torch.isnan(tensor).sum(dim=0, dtype=torch.float)

#     # Replace NaN values with 0
#     tensor = tensor.masked_fill(torch.isnan(tensor), 0)

#     # Take the sum of each column
#     row_sums = tensor.sum(dim=0)


#     # Divide the sum of each cateogry by the number of rows minus the number of NaN values
#     processed_tensor = row_sums / (tensor.shape[0] - nan_count)

#     return processed_tensor


# # Create a 2D tensor with some NaN values
# mious = []
# a = torch.tensor([0.3763, 0.4762, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,    float('nan'),
#         0.6402, 0.0000, 0.5277,    float('nan'), 0.8976, 0.0000, 0.7143, 0.0000, 0.7752,
#         0.3821, 0.0000])

# b = torch.tensor([0.3665, 0.4225, 0.0000, 0.0000,    float('nan'), 0.0000, 0.0000, 0.0000,    float('nan'),
#         0.6716, 0.0000, 0.5279,    float('nan'), 0.8864, 0.0000, 0.7052, 0.0000, 0.7460,
#         0.3504, 0.0000])

# mious.append(a)
# mious.append(b)

# result = torch.cat(mious, dim=-1).reshape(-1, 20)

# print('result: \n', result)



# # Apply the nan_process() function to the tensor
# processed_tensor = nan_process(result)

# print('processed_tensor: \n', processed_tensor)  

# def calc_iou(pred, target, num_classes):
#     # Initialize a tensor to store the IoU for each class
#     iou_per_class = torch.zeros(num_classes)

#     # Iterate over each class
#     for c in range(1, num_classes):
#         # Create masks that identify the predicted and target elements for class c
#         pred_mask = (pred == c)
#         target_mask = (target == c)

#         # Compute the area of the intersection of the predicted and target tensors for class c
#         intersection = (pred_mask * target_mask).sum()

#         # Compute the area of the union of the predicted and target tensors for class c
#         union = pred_mask.sum() + target_mask.sum() - intersection

#         # Compute the IoU for class c as the ratio of the intersection to the union
#         iou_per_class[c] = intersection / union

#     # Return the IoU tensor
#     return iou_per_class


# prediction = torch.tensor([[4, 2, 1], [2, 3, 1], [5, 2, 1], [ 1, 2, 3]])
# target = torch.tensor([0, 2, 1, 2])

# # print('pred[:, 1:] \n', pred[:, 1:])

# # print('pred[:, 1:].argmax(dim=-1) \n', pred[:, 1:].argmax(dim=-1))
# def calc_iou_original(pred, target, num_classes):
#     # Initialize a tensor to store the IoU for each class
#     iou_per_class = torch.zeros(num_classes)

# #     pred = pred.argmax(-1)
#     print(pred)
#     # Iterate over each class
#     for c in range(1, num_classes):
#         # Create masks that identify the predicted and target elements for class c
#         pred_mask = (pred == c)
#         target_mask = (target == c)

#         # Compute the area of the intersection of the predicted and target tensors for class c
#         intersection = (pred_mask * target_mask).sum()

#         # Compute the area of the union of the predicted and target tensors for class c
#         union = pred_mask.sum() + target_mask.sum() - intersection

#         # Compute the IoU for class c as the ratio of the intersection to the union
#         iou_per_class[c] = intersection / union

#     # Return the IoU tensor
#     return iou_per_class
# # out = calc_iou(pred[:, 1:].argmax(dim=-1), target, num_classes=4)
# # print('Out: \n', out)

# def calc_iou(pred, target, num_classes, ignore_index):
#     # Initialize a tensor to store the IoU for each class
#     iou_per_class = torch.zeros(num_classes)

#     pred[:, ignore_index] = torch.tensor([-float('inf')])
#     pred = pred.argmax(-1)
#     print(pred)
#     # Iterate over each class
#     for c in range(num_classes) :
#         if c == ignore_index:
#                 continue
#         else:
                        
#                 # Create masks that identify the predicted and target elements for class c
#                 pred_mask = (pred == c)
#                 target_mask = (target == c)

#                 # Compute the area of the intersection of the predicted and target tensors for class c
#                 intersection = (pred_mask * target_mask).sum()

#                 # Compute the area of the union of the predicted and target tensors for class c
#                 union = pred_mask.sum() + target_mask.sum() - intersection

#                 # Compute the IoU for class c as the ratio of the intersection to the union
#                 iou_per_class[c] = intersection / union

#     # Return the IoU tensor
#     return iou_per_class

# print(prediction.size(), target.size())

# # print(prediction.argmax(-1))

# out = calc_iou(prediction, target, num_classes=3, ignore_index=0)
# print('Out: \n', out)

# prediction_modified = torch.tensor([1, 1, 1, 2])
# out = calc_iou_original(prediction_modified, target, num_classes=3)
# print('Should be Out: \n', out)

# from utils import utils

# DATA_path = './semantic-kitti.yaml' # for running in docker


# utils.get_xentropy_class_string(0, DATA_path)

import torch

import torch.nn.functional as F

points = torch.randn(5, 2)

out = F.pad(points, (0, 0, 0, 5), "constant", 0)

print(f'points: {points}', '\n')
print(f'out: {out}', '\n')
print(f'out.size: {out.size()}')

