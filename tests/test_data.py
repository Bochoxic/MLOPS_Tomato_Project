import sys
sys.path.append("./src/data/")

import os

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tomatoDataset import tomatoDataset


@pytest.mark.skipif(not os.path.exists("./data"), reason="Data files not found")
def test_data():
    # Defend directories to load data
    data_dir = 'data/processed'

    # Load train dataset and create train data loader
    train_dataset = tomatoDataset('train', data_dir)
    train_loader = DataLoader(train_dataset, batch_size=1)
    # Load test dataset and create test data loader
    test_dataset = tomatoDataset('valid', data_dir)
    test_loader = DataLoader(test_dataset,  batch_size=1)


    for images, labels in train_loader:
        assert list(images.shape) == [1, 3, 256, 256], 'Images shape should be [batch_size, 1, 256, 256]'
        assert len(labels) == 1, 'Labels size should be batch_size'
    
    for images, labels in test_loader: 
        assert list(images.shape) == [1, 3, 256, 256], 'Images shape should be [batch_size, 1, 256, 256]'
        assert len(labels) == 1, 'Labels size should be batch_size'

    assert len(train_dataset) == 25850, 'Train dataset length should be 25850'
    assert len(test_dataset) == 6681, 'Test dataset length should be 6681'

# Run test data
test_data()