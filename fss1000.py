import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import cv2


class FSS1000(Dataset):
    def __init__(self, root='data', image_set='train', h=512, w=512):
        self.root = root
        self.h = h
        self.w = w
        fss1000_dir = os.path.join(self.root, 'fewshot_data')

        self.classes_name = ['background'] + os.listdir(fss1000_dir)
        self.classes = [0] + list(range(1,len(self.classes_name)))

        if image_set == 'train':
            limits = (1, 6)
        elif image_set == 'val':
            limits = (6, 11)
        self.dataset = []
        for i in range(1, len(self.classes_name)):
            class_dir = os.path.join(fss1000_dir, self.classes_name[i])
            for j in range(limits[0], limits[1]):
                input_path = os.path.join(class_dir, f'{j}.jpg')
                target_path = os.path.join(class_dir, f'{j}.png')
                class_label = self.classes[i]
                self.dataset.append((input_path, target_path, class_label))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_path, target_path, class_label = self.dataset[idx]

        img = np.array(Image.open(input_path))

        if self.h and self.w:
            img = cv2.resize(img, (self.w, self.h), cv2.INTER_NEAREST)

        target = Image.open(target_path)
        target = np.array(target)
        target = np.min(target , axis=2)
        target = target/255*class_label
        if self.h and self.w:
            target = cv2.resize(target, (self.w, self.h), cv2.INTER_NEAREST)

        return torch.Tensor(img).transpose(2, 1).transpose(0, 1), torch.Tensor(target)

