# DeeplabV3
Reimplementation of deeplabv3 for Image Segmentation course using torchvision 

## Details on Project
A review on DeeplabV3 and how well it performs without any data augmentation (i.e. only how well the architecture will perform)

Note that even though cityscapes is implemented, no training was done on it due to resource constraints. The code has not been fully run due to that and can be prone to bugs.

## Directory Structure
```
chkpt/
    voc-epoch100.pt
data/
    VOCdevkit/
        ...
fss1000plots/
vocplots/
    epoch1-val-segmentation.png
    ...
.gitignore
cityscapes.py
deeplabv3.py
fss1000.py
fssaccuracy.py
README.md
train.py
visualisefss.py
visualisevoc.py
voc.py
vocaccuracy.py
```

## Models
Pretrained models are available at https://drive.google.com/drive/folders/16DyapBgw4mqJ0QmOyk8TPs0MUBBDg2gC?usp=sharing

## Training
To train new model
```
$ python train.py
```
Parameters can be set in the file to toggle between different datasets

## Evaluation and Visualisation
To evalute models
```
$ python vocaccuracy.py
$ python fssaccuracy.py
```

To visualise segmentation
```
$ python visualisevoc.py
$ python visualisefss.py
```
Visualisation can be toggled to see the best segmentations and the worst