import opendatasets as od
import numpy as np
import os
import torch
import wandb
import torch.nn as nn
import tqdm as tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torch.autograd import Variable
from torchsummary import summary
import timm
from torch import optim
from model_ffnn import Net
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
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
    wandb.init(project="Tomato Project")
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
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    test_transformer = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    trainloader = DataLoader(torchvision.datasets.ImageFolder(project_path+'/data/raw/tomato-disease-multiple-sources/train',
     transform=train_transformer),batch_size=hparams["batch_size"],shuffle=True)
    testloader = DataLoader(torchvision.datasets.ImageFolder(project_path+'/data/raw/tomato-disease-multiple-sources/valid',
     transform=test_transformer),batch_size=hparams["batch_size"],shuffle=True)

    model = Net(hparams["lr"])
    wandb.watch(model, log_freq=100)
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
        wandb.log({"loss": loss})


def train_lightning(config, device):
    # Load hyperparameters from hydra
    hparams = config.experiment.hyperparams #load the hyperparameters. config/experiment/exp1.yaml --> hyperparams

    # Define transformations
    train_transformer = transforms.Compose([
    transforms.Resize((10,10)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    test_transformer = transforms.Compose([
        transforms.Resize((10,10)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    # Create dataloader and testloader
    trainloader = DataLoader(torchvision.datasets.ImageFolder(project_path+'/data/raw/tomato-disease-multiple-sources/train',
     transform=train_transformer),batch_size=hparams["batch_size"],shuffle=True)
    testloader = DataLoader(torchvision.datasets.ImageFolder(project_path+'/data/raw/tomato-disease-multiple-sources/valid',
     transform=test_transformer),batch_size=hparams["batch_size"],shuffle=True)
 
    # Define early stopping callback: Stop training if validation loss doesn't improve during "patience" epochs 
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")
    # Create trainer
    wandb.finish()
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=hparams["n_epoch"], limit_train_batches=hparams["limit_batches"], callbacks=[early_stopping_callback], accelerator='gpu', logger=wandb_logger)
    # Define model
    model = Net(hparams["lr"])
    wandb.watch(model, log_freq=100)
    # Train model
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)
    wandb.finish()
  

# Run training
main()