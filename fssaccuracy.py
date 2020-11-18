import torch
from torch.utils.data import DataLoader
from fss1000 import FSS1000
import numpy as np


voc_checkpoint_dir = 'chkpt/fss1000-epoch90.pt'
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
num_classes = 1001

def intersect_and_union(pred, target, num_class):
    '''
    References https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/utils.py
    '''
    pred = np.asarray(pred, dtype=np.uint8).copy()
    target = np.asarray(target, dtype=np.uint8).copy()

    # 255 -> 0
    pred += 1
    target += 1
    pred = pred * (target > 0)

    inter = pred * (pred == target)
    (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
    (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
    (area_target, _) = np.histogram(target, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_target - area_inter

    return (area_inter, area_union)

def calculate_accuracy(loader, model, device):
    model.eval()
    total_intersect = np.array([0] * num_classes)
    total_union = np.array([0] * num_classes)
    with torch.no_grad():
        for idx, (data, target) in enumerate(loader):
            data = data.to(device)

            output = model(data)
            output_predictions = output.argmax(1).detach().cpu()
            intersect, union = intersect_and_union(output_predictions, target, num_classes)
            total_intersect += intersect
            total_union += union
            
    total_union[total_union == 0] = 1
    iou = (total_intersect/total_union)
    avg_iou = np.mean(iou)
    return avg_iou



if __name__ == '__main__':
    model = torch.load(voc_checkpoint_dir)

    val_data = FSS1000('data/', image_set='val', h=224, w=224)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    avg_iou = calculate_accuracy(val_loader, model, device)
    print(f'IOU: {avg_iou*100}')