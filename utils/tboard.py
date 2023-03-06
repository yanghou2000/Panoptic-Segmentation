from torch.utils.tensorboard import SummaryWriter
from utils import utils

def add_list(writer,values,epoch, DATA_path, name):
    for ncls in range(len(values)):
        # write class number to class str function
        class_name = utils.get_xentropy_class_string(ncls, DATA_path)

        # by defualt label 0 is ignored, so iou for label 0 should be 0
        writer.add_scalar(f'{name}/{ncls}_{class_name}', values[ncls], epoch)