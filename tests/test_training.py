import sys

sys.path.append('./src/data')
sys.path.append('./src/models')

import os.path

import pytest
import pytorch_lightning as pl
import torch
from model import Net
from tomatoDataset import tomatoDataset
from torch.utils.data import DataLoader
from torchvision import transforms


@pytest.mark.skipif(not os.path.exists("./data"), reason="Data files not found")
def test_training():
    # Defend directories to load data
    data_dir = 'data/processed'

    # Load train dataset and create train data loader
    train_dataset = tomatoDataset('train', data_dir)
    train_loader = DataLoader(train_dataset, batch_size=1)
    # Load test dataset and create test data loader
    test_dataset = tomatoDataset('valid', data_dir)
    test_loader = DataLoader(test_loader, data_dir)
 
    # Create trainer
    trainer = pl.Trainer(max_epochs=1, limit_train_batches=0.05)
    # Define model
    model = Net()
    # Train model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    train_loss = trainer.logged_metrics['train_loss']
    val_loss = trainer.logged_metrics['val_loss']
    val_acc = trainer.logged_metrics['val_acc']

    assert train_loss >= 0, 'Training loss should be >= 0'
    assert val_loss >= 0, 'Validation loss should be >= 0'
    assert val_acc <= 1, 'Validation accuracy should be <= 1'
