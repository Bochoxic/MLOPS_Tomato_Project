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
from model import Net

def train():
    train_transformer = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],
                        [0.5,0.5,0.5])
    ])

    test_transformer = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],
                            [0.5,0.5,0.5])
    ])

    trainloader = DataLoader(torchvision.datasets.ImageFolder("data/raw/tomato-disease-multiple-sources/train",transform=train_transformer),batch_size=32,shuffle=True)
    testloader = DataLoader(torchvision.datasets.ImageFolder("data/raw/tomato-disease-multiple-sources/valid",transform=test_transformer),batch_size=32,shuffle=True)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

        
    n_epoch = 20
    for epoch in range (n_epoch):
        loss_tracker = []
        contador=0
        for images, labels in trainloader:
            print(contador)
            optimizer.zero_grad()
            log_ps = model(images)
            #print(log_ps)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())
            contador+=1


        print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")
train()