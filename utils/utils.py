import torch

def calc_iou(pred, target, num_classes):
    # Initialize a tensor to store the IoU for each class
    iou_per_class = torch.zeros(num_classes)

    # Iterate over each class
    for c in range(num_classes):
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

def calc_miou(ious,num_classes):
    mask = torch.isnan(ious)
    # print(mask.nonzero())
    tnc = torch.tensor(num_classes) - torch.sum(mask)
    ious = ious.masked_fill(mask,0)
    siou = torch.sum(ious)
    # print(tnc,siou)
    return siou/tnc