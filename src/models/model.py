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

import pytorch_lightning as pl


class Net(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()

        self.model = timm.create_model('resnet18',pretrained=True, num_classes=11)

        self.lr = lr
        self.dropout = nn.Dropout(p=0.3)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx): 
        images, labels = batch
        preds = self(images)
        loss = self.criterion(preds, labels)
        self.log("train_loss", float(loss))

        return loss

    def configure_optimizers(self): 
        return optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.criterion(preds, labels)
        # Calculate predictions probabilities
        ps = torch.exp(preds)
        # Most likely classes 
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        # Get model accuracy
        accuracy = torch.mean(equals.type(torch.FloatTensor))

        # Log metrics
        self.log("val_loss", float(loss))
        self.log("val_acc", float(accuracy))
