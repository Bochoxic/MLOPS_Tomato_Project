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
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

# Define logger
log = logging.getLogger(__name__)

print = log.info #print --> log.info : in this way hydra save prints in output folder 

project_path=os.getcwd() #so hydra doesn't change the path 

# Load config file with hydra
@hydra.main(config_path="../../config", config_name='default_config.yaml') #config/default_config.yaml
def main(config): 
    # Use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using torch device of type {device.type}{": " + torch.cuda.get_device_name(device) if device.type == "cuda" else ""}')
    print("Start Training...")

    # Select between normal or lightning training (config/experiment/exp1/training: lightning or normal)
    if config.experiment.training == 'lightning': 
        train_lightning(config, device)
    else:    
        train(config, device)

def train(config, device):
    
    hparams = config.experiment.hyperparams #load the hyperparameters. config/experiment/exp1.yaml --> hyperparams

    train_transformer = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    test_transformer = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    trainloader = DataLoader(torchvision.datasets.ImageFolder(project_path + '/data/raw/tomato-disease-multiple-sources/train', transform=train_transformer),batch_size=hparams["batch_size"],shuffle=True)
    testloader = DataLoader(torchvision.datasets.ImageFolder(project_path  + '/data/raw/tomato-disease-multiple-sources/valid', transform=test_transformer),batch_size=hparams["batch_size"],shuffle=True)

    model = Net()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])

    for epoch in range (hparams["n_epoch"]):
        loss_tracker = []
        contador=0
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            print(f"Batch: {contador}/{len(trainloader)}")
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())
            contador+=1
        

        print(f"Epoch {epoch+1} /{hparams['n_epoch']}. Loss: {loss}")


def train_lightning(config, device):
    # Load hyperparameters from hydra
    hparams = config.experiment.hyperparams #load the hyperparameters. config/experiment/exp1.yaml --> hyperparams

    # Define transformations
    train_transformer = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    test_transformer = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    # Create dataloader and testloader
    trainloader = DataLoader(torchvision.datasets.ImageFolder(project_path + '/data/raw/tomato-disease-multiple-sources/train', transform=train_transformer),batch_size=hparams["batch_size"],shuffle=True)
    testloader = DataLoader(torchvision.datasets.ImageFolder(project_path  + '/data/raw/tomato-disease-multiple-sources/valid', transform=test_transformer),batch_size=hparams["batch_size"],shuffle=True)
 
    # Define early stopping callback: Stop training if validation loss doesn't improve during "patience" epochs 
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")
    # Create trainer
    trainer = pl.Trainer(max_epochs=10, limit_train_batches=0.05, callbacks=[early_stopping_callback], accelerator='gpu')
    # Define model
    model = Net()
    # Train model
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)
  

# Run training
main()