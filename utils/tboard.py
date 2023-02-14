from torch.utils.tensorboard import SummaryWriter
from utils import utils

def add_list(writer,ious,epoch, DATA_path):
    for ncls in range(len(ious)):
        # write class number to class str function
        class_name = utils.get_xentropy_class_string(ncls, DATA_path)

        # by defualt label 0 is ignored, so iou for label 0 should be 0
        writer.add_scalar(f'IoU_{ncls}_{class_name}', ious[ncls], epoch)