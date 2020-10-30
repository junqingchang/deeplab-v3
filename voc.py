import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import cv2


class VOCSegmentation(Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
    """

    def __init__(
            self,
            root: str,
            image_set: str = "train",
            h = 512,
            w = 512
    ):
        super(VOCSegmentation, self).__init__()
        self.year = "2012"
        valid_sets = ["train", "trainval", "val"]

        self.root = root
        self.h = h
        self.w = w

        base_dir = os.path.join('VOCdevkit', 'VOC2012')
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = np.array(Image.open(self.images[index]))
        img = cv2.resize(img, (self.w, self.h), cv2.INTER_NEAREST)

        target = Image.open(self.masks[index])
        target = np.array(target)
        idx255 = target == np.ones_like(target) * 255
        target[idx255] = 0
        target = cv2.resize(target, (self.w, self.h), cv2.INTER_NEAREST)

        return torch.Tensor(img).transpose(2, 1).transpose(0, 1), torch.Tensor(target)


    def __len__(self):
        return len(self.images)