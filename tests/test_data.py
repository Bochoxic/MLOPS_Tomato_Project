import sys
sys.path.append("./src/data/")

import os

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision


@pytest.mark.skipif(not os.path.exists("./data"), reason="Data files not found")
def test_data():
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
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    # Create dataloader and testloader
    train_loader = DataLoader(torchvision.datasets.ImageFolder('data/raw/tomato-disease-multiple-sources/train',
     transform=train_transformer),batch_size=bs,shuffle=True, drop_last=True)
    test_loader = DataLoader(torchvision.datasets.ImageFolder('data/raw/tomato-disease-multiple-sources/valid',
     transform=test_transformer),batch_size=bs, drop_last=True)


    for images, labels in test_loader: 
        assert list(images.shape) == [bs, 3, 256, 256], 'Images shape should be [batch_size, 3, 256, 256]'
        assert len(labels) == bs, 'Labels size should be batch_size'

    for images, labels in train_loader:
        assert list(images.shape) == [bs, 3, 256, 256], 'Images shape should be [batch_size, 3, 256, 256]'
        assert len(labels) == bs, 'Labels size should be batch_size'

    print('TEST DATA DONE!!!!!!!!!!!!!!!!!!!!!!')
    
    
# Run test data
test_data()