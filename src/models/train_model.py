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

import logging
import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
print = log.info #print --> log.info


project_path=os.getcwd() #so hydra doesn't change the path 

@hydra.main(config_path="../../config", config_name='default_config.yaml') #config/default_config.yaml
def train(config: DictConfig):
    #print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    logger = logging.getLogger(__name__)
    logger.info("Start Training...")
    
    hparams = config.experiment.hyperparams #load the hyperparameters. config/experiment/exp1.yaml --> hyperparams

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

    trainloader = DataLoader(torchvision.datasets.ImageFolder(project_path + '/data/raw/tomato-disease-multiple-sources/train', transform=train_transformer),batch_size=hparams["batch_size"],shuffle=True)
    testloader = DataLoader(torchvision.datasets.ImageFolder(project_path  + '/data/raw/tomato-disease-multiple-sources/valid', transform=test_transformer),batch_size=hparams["batch_size"],shuffle=True)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])

       
    for epoch in range (hparams["n_epoch"]):
        loss_tracker = []
        #contador=0
        for images, labels in trainloader:
            #print(contador)
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())
            #contador+=1
        

        print(f"Epoch {epoch+1} /{hparams['n_epoch']}. Loss: {loss}")
train()