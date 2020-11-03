import torch
import torch.nn as nn
from torch.utils.data import dataset
from voc import VOCSegmentation
from cityscapes import Cityscapes
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from deeplabv3 import DeepLabV3
import sys
import random
from PIL import Image
import os


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
NUM_CLASSES = 21 #21 for VOC, 35 for Cityscapes
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
BATCH_SIZE = 8
SAVE_MODEL_EVERY = 10

checkpoint_dir = 'chkpt'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

dataset_type = 'voc'
plot_dir = f'{dataset_type}plots/'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

resume_training = False


def train(loader, model, criterion, optimizer, device, print_every=50):
    model.train()
    train_loss = 0
    for idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target.long())

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
        if idx % print_every == 0:
            print(f'{idx+1}/{len(loader)} Loss: {loss.item()}')
        
    return train_loss/len(loader)

def eval(loader, model, criterion, device):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target.long())

            eval_loss += loss.item()

    return eval_loss/len(loader)

def display_segmentation(dataset, model, img_path, device):
    model.cpu()
    model.eval()

    sample_data = dataset[random.randint(0, len(dataset)-1)]
    sample = sample_data[0]
    sample_output = sample_data[1]
    with torch.no_grad():
        output = model(sample.unsqueeze(0))[0]
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(sample.size()[1:])
    r.putpalette(colors)

    s = Image.fromarray(sample_output.byte().cpu().numpy()).resize(sample.size()[1:])
    s.putpalette(colors)

    plt.subplot(1, 3, 1)
    plt.imshow(sample.transpose(0, 1).transpose(1, 2).long())
    plt.subplot(1, 3, 2)
    plt.imshow(r)
    plt.subplot(1, 3, 3)
    plt.imshow(s)

    plt.savefig(img_path)
    plt.close()

    model.to(device)

def display_segmentation_cityscapes(dataset, model, img_path, device):
    model.cpu()
    model.eval()

    sample_data = dataset[random.randint(0, len(dataset)-1)]
    sample = sample_data[0]
    sample_output = sample_data[1]
    with torch.no_grad():
        output = model(sample.unsqueeze(0))[0]
    output_predictions = output.argmax(0)

    plt.subplot(1, 3, 1)
    plt.imshow(sample.transpose(0, 1).transpose(1, 2).long())
    plt.subplot(1, 3, 2)
    plt.imshow(sample_output)
    plt.subplot(1, 3, 3)
    plt.imshow(output_predictions.long())

    plt.savefig(img_path)
    plt.close()

    model.to(device)


if __name__ == '__main__':
    if dataset_type == 'voc':
        train_data = VOCSegmentation('data/')
        val_data = VOCSegmentation('data/', image_set='val',)
    elif dataset_type == 'cityscapes':
        train_data = Cityscapes('data/', h=256, w=512)
        val_data = Cityscapes('data/', split='val', h=256, w=512)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = DeepLabV3(num_classes=NUM_CLASSES)

    if resume_training:
        if dataset_type == 'voc':
            model = torch.load(f'{checkpoint_dir}/voc.pt')
        elif dataset_type == 'cityscapes':
            model = torch.load(f'{checkpoint_dir}/cityscapes.pt')

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)


    train_losses = []
    val_losses = []

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - (epoch / EPOCHS)) ** 0.9)


    for epoch in range(1, EPOCHS+1):
        train_loss = train(train_loader, model, criterion, optimizer, device)
        val_loss = eval(val_loader, model, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')

        display_segmentation(val_data, model, f'{plot_dir}/epoch{epoch}-val-segmentation.png', device)

        if epoch % SAVE_MODEL_EVERY == 0:
            if dataset_type == 'voc':
                torch.save(model, f'{checkpoint_dir}/voc-epoch{epoch}.pt')
            elif dataset_type == 'cityscapes':
                torch.save(model, f'{checkpoint_dir}/cityscapes-epoch{epoch}.pt')

        plt.figure()
        plt.title('Train Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(train_losses)
        plt.savefig(f'{plot_dir}/train_loss.png')
        plt.close()

        plt.figure()
        plt.title('Val Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(val_losses)
        plt.savefig(f'{plot_dir}/val_loss.png')
        plt.close()

        scheduler.step()

    if dataset_type == 'voc':
        torch.save(model, f'{checkpoint_dir}/voc-completed.pt')
    elif dataset_type == 'cityscapes':
        torch.save(model, f'{checkpoint_dir}/cityscapes-completed.pt')
    print(f'Train Complete')

    

