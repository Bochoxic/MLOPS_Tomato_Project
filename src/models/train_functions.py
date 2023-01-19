import torchvision
import wandb
import torch

import torchvision.transforms as transforms
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from model import Net_ffnn, Net
from torch import optim
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger



def prepare_data(project_path, batch_size, image_size_x, image_size_y):

    train_transformer = transforms.Compose([
    transforms.Resize((image_size_x,image_size_y)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    test_transformer = transforms.Compose([
        transforms.Resize((image_size_x,image_size_y)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    trainloader = DataLoader(torchvision.datasets.ImageFolder(project_path+'/data/raw/tomato-disease-multiple-sources/train',
     transform=train_transformer),batch_size=batch_size,shuffle=True)
    testloader = DataLoader(torchvision.datasets.ImageFolder(project_path+'/data/raw/tomato-disease-multiple-sources/valid',
     transform=test_transformer),batch_size=batch_size,shuffle=True)

    return trainloader, testloader

def train(project_path, image_size_x, image_size_y, device, model_type, batch_size, lr, n_epoch):
    
    trainloader, _ = prepare_data(project_path, batch_size, image_size_x= image_size_x, image_size_y= image_size_y)

    if model_type == "fast":
        model = Net_ffnn(lr)
    elif model_type == "precise":
        model = Net(lr)

    wandb.watch(model, log_freq=100)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range (n_epoch):
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
        

        print(f"Epoch {epoch+1} /{n_epoch}. Loss: {loss}")
        wandb.log({"loss": loss})

    torch.save(model.state_dict(), project_path + "/models/trained_model.pt")


def train_lightning(project_path, image_size_x, image_size_y, model_type, batch_size, lr, n_epoch, limit_batches, patience):

    trainloader, testloader = prepare_data(project_path, batch_size, image_size_x= image_size_x, image_size_y= image_size_y)
 
    # Define early stopping callback: Stop training if validation loss doesn't improve during "patience" epochs 
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=patience, verbose=True, mode="min")
    early_stopping_threshold = EarlyStopping(monitor="val_loss", divergence_threshold=10)
    checkpoint_callback = ModelCheckpoint(dirpath=project_path + "/models/lightning", save_top_k=3, monitor="val_loss")
    # Create trainer
    wandb.finish() # Finsih previous wandb process
    wandb_logger = WandbLogger() # Initialize the new wandb
    trainer = pl.Trainer(callbacks=[early_stopping_callback, checkpoint_callback, early_stopping_threshold], accelerator='auto', logger=wandb_logger, max_epochs=n_epoch)
    # Define model

    if model_type == "fast":
        model = Net_ffnn(lr=lr)
    elif model_type == "precise":
        model = Net(lr=lr)
        
    wandb.watch(model, log_freq=100)    
    # Train model
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)
    checkpoint_callback.best_model_path
    wandb.finish()
    torch.save(model.state_dict(), project_path + "/models/trained_model.pt")
