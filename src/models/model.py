import timm 
import opendatasets as od
import numpy as np
import os
import torch
import torch.nn as nn
import tqdm as tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torch.autograd import Variable
from torchsummary import summary
import timm
from torch import optim
#import sklearn.metrics as metrics


def Net():
    model = timm.create_model('resnet18',pretrained=True, num_classes=11)
    return model



