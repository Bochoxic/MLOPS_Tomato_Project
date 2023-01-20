import sys

sys.path.append('./src/data')
sys.path.append('./src/models')

import os.path

import pytest
import pytorch_lightning as pl
import torch
from model import Net
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision


@pytest.mark.skipif(not os.path.exists("./data"), reason="Data files not found")
def test_training():
    # Defend directories to load data
    data_dir = 'data/processed'
    bs = 32
    # Define transformations
    train_transformer = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    test_transformer = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    # Create dataloader and testloader
    train_loader = DataLoader(torchvision.datasets.ImageFolder('data/raw/tomato-disease-multiple-sources/train',
     transform=train_transformer),batch_size=bs,shuffle=True)
    test_loader = DataLoader(torchvision.datasets.ImageFolder('data/raw/tomato-disease-multiple-sources/valid',
     transform=test_transformer),batch_size=bs)
 
    # Create trainer
    trainer = pl.Trainer(max_epochs=1, limit_train_batches=0.0013)
    # Define model
    model = Net(lr=0.0001)
    # Train model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    train_loss = trainer.logged_metrics['train_loss']
    val_loss = trainer.logged_metrics['val_loss']
    val_acc = trainer.logged_metrics['val_acc']

    assert train_loss >= 0, 'Training loss should be >= 0'
    assert val_loss >= 0, 'Validation loss should be >= 0'
    assert val_acc <= 1, 'Validation accuracy should be <= 1'

    print('TEST TRAINING DONE!!!!!!!!!!!!!!!!!!!!!!!')

test_training()