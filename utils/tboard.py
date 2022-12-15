from torch.utils.tensorboard import SummaryWriter

def add_iou(writer,ious,epoch):
    for ncls in range(len(ious)):
        # TODO: write class number to class str function
        # class_name = map_i2n(ncls)
        # writer.add_scalar(f'IoU_{class_name}',ious,epoch)
    
        writer.add_scalar(f'IoU_{ncls}', ious, epoch)