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