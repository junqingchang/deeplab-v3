import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


if __name__ == '__main__':
    train_data = VOCSegmentation('data/', download=True)
    val_data = VOCSegmentation('data/', image_set='trainval', download=True)
    test_data = VOCSegmentation('data/', image_set='val', download=True)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

    print(train_data[0][0].shape)
    print(train_data[0][1].shape)
    print(train_data[1][0].shape)
    print(train_data[1][1].shape)
