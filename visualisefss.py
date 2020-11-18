import torch
from fss1000 import FSS1000
import random
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np


voc_checkpoint_dir = 'chkpt/fss1000-epoch90.pt'
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
num_classes = 1001
display_best = False

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


if __name__ == '__main__':
    model = torch.load(voc_checkpoint_dir)
    model.to(device).eval()
    val_data = FSS1000('data/', image_set='val', h=224, w=224)

    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    iou_values = {}

    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            data = data.to(device)
            output = model(data)

            output_predictions = output.argmax(1).detach().cpu()
            intersect, union = intersect_and_union(output_predictions, target, num_classes)
            union[union == 0] = 1
            iou = intersect/union
            avg_iou = np.mean(iou)
            iou_values[avg_iou] = idx

    sorted_values = sorted(iou_values, reverse=display_best)

    sample_data = val_data[iou_values[sorted_values[0]]]
    sample = sample_data[0].to(device)
    sample_output = sample_data[1]
    with torch.no_grad():
        output = model(sample.unsqueeze(0))[0]
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy())
    r.putpalette(colors)

    s = Image.fromarray(sample_output.byte().cpu().numpy())
    s.putpalette(colors)

    plt.subplot(3, 3, 1)
    plt.title('Original')
    plt.imshow(sample.detach().cpu().transpose(0, 1).transpose(1, 2).long())
    plt.subplot(3, 3, 2)
    plt.title('Prediction')
    plt.imshow(r)
    plt.subplot(3, 3, 3)
    plt.title('Target')
    plt.imshow(s)

    sample_data = val_data[iou_values[sorted_values[1]]]
    sample = sample_data[0].to(device)
    sample_output = sample_data[1]
    with torch.no_grad():
        output = model(sample.unsqueeze(0))[0]
    output_predictions = output.argmax(0)

    r = Image.fromarray(output_predictions.byte().cpu().numpy())
    r.putpalette(colors)

    s = Image.fromarray(sample_output.byte().cpu().numpy())
    s.putpalette(colors)

    plt.subplot(3, 3, 4)
    plt.title('Original')
    plt.imshow(sample.detach().cpu().transpose(0, 1).transpose(1, 2).long())
    plt.subplot(3, 3, 5)
    plt.title('Prediction')
    plt.imshow(r)
    plt.subplot(3, 3, 6)
    plt.title('Target')
    plt.imshow(s)

    sample_data = val_data[iou_values[sorted_values[2]]]
    sample = sample_data[0].to(device)
    sample_output = sample_data[1]
    with torch.no_grad():
        output = model(sample.unsqueeze(0))[0]
    output_predictions = output.argmax(0)

    r = Image.fromarray(output_predictions.byte().cpu().numpy())
    r.putpalette(colors)

    s = Image.fromarray(sample_output.byte().cpu().numpy())
    s.putpalette(colors)

    plt.subplot(3, 3, 7)
    plt.title('Original')
    plt.imshow(sample.detach().cpu().transpose(0, 1).transpose(1, 2).long())
    plt.subplot(3, 3, 8)
    plt.title('Prediction')
    plt.imshow(r)
    plt.subplot(3, 3, 9)
    plt.title('Target')
    plt.imshow(s)

    plt.show()