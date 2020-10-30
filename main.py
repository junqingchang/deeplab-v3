import torch
import torch.nn as nn
from voc import VOCSegmentation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from deeplabv3 import DeepLabV3
import sys


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
NUM_CLASSES = 21
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
EPOCHS = 100
BATCH_SIZE = 8

voc_checkpoint_dir = 'chkpt/voc.pt'
plot_dir = 'plots/'


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
    sample_input, sample_target = dataset[0]
    with torch.no_grad():
        model_output = model(sample_input.unsqueeze(0))
    predicted_seg = torch.argmax(model_output, dim=1)

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_input.transpose(0, 1).transpose(1, 2).long())
    plt.subplot(1, 3, 2)
    plt.imshow(sample_target)
    plt.clim(0,20)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_seg.squeeze())
    plt.clim(0,20)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(img_path)
    plt.close()

    model.to(device)
    

if __name__ == '__main__':
    train_data = VOCSegmentation('data/')
    val_data = VOCSegmentation('data/', image_set='val',)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = DeepLabV3(num_classes=NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    best_loss = sys.maxsize

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

        if val_loss < best_loss:
            best_loss = val_loss
            print('Better model, saving new model!')
            torch.save(model, voc_checkpoint_dir)

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

    print(f'Train Complete')

    

