import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ASPP(nn.Module):
  '''
  References https://github.com/AvivSham/DeepLabv3 for ASPP module
  '''
  def __init__(self, in_channels, out_channels = 256):
    super(ASPP,self).__init__()
    
    
    self.relu = nn.ReLU(inplace=True)
    
    self.conv1 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          padding = 0,
                          dilation=1,
                          bias=False)
    
    self.bn1 = nn.BatchNorm2d(out_channels)
    
    self.conv2 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride=1,
                          padding = 6,
                          dilation = 6,
                          bias=False)
    
    self.bn2 = nn.BatchNorm2d(out_channels)
    
    self.conv3 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride=1,
                          padding = 12,
                          dilation = 12,
                          bias=False)
    
    self.bn3 = nn.BatchNorm2d(out_channels)
    
    self.conv4 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride=1,
                          padding = 18,
                          dilation = 18,
                          bias=False)
    
    self.bn4 = nn.BatchNorm2d(out_channels)
    
    self.conv5 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          stride=1,
                          padding = 0,
                          dilation=1,
                          bias=False)
    
    self.bn5 = nn.BatchNorm2d(out_channels)
    
    self.convf = nn.Conv2d(in_channels = out_channels * 5, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          stride=1,
                          padding = 0,
                          dilation=1,
                          bias=False)
    
    self.bnf = nn.BatchNorm2d(out_channels)
    
    self.adapool = nn.AdaptiveAvgPool2d(1)  
   
  
  def forward(self,x):
    
    x1 = self.conv1(x)
    x1 = self.bn1(x1)
    x1 = self.relu(x1)
    
    x2 = self.conv2(x)
    x2 = self.bn2(x2)
    x2 = self.relu(x2)
    
    x3 = self.conv3(x)
    x3 = self.bn3(x3)
    x3 = self.relu(x3)
    
    x4 = self.conv4(x)
    x4 = self.bn4(x4)
    x4 = self.relu(x4)
    
    x5 = self.adapool(x)
    x5 = self.conv5(x5)
    x5 = self.bn5(x5)
    x5 = self.relu(x5)
    x5 = F.interpolate(x5, size = tuple(x4.shape[-2:]), mode='bilinear', align_corners=True)
    
    x = torch.cat((x1,x2,x3,x4,x5), dim = 1)
    x = self.convf(x)
    x = self.bnf(x)
    x = self.relu(x)
    
    return x


class DeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3, self).__init__()
        self.num_classes = num_classes
        self.backbone = models.resnet50(pretrained=True)
        self.aspp = ASPP(in_channels=1024)

        self.conv = nn.Conv2d(in_channels = 256, out_channels = self.num_classes,
                          kernel_size = 1, stride=1, padding=0)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.resnet_feature_extract(self.backbone, x)
        x = self.aspp(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x

    def resnet_feature_extract(self, resnet, x):
        x = resnet.conv1(x)
        x = resnet.bn1(x)
        x = resnet.relu(x)
        x = resnet.maxpool(x)

        x = resnet.layer1(x)
        x = resnet.layer2(x)
        x = resnet.layer3(x)
        return x
