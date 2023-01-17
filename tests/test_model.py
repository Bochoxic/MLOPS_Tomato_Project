import sys
sys.path.append('./src/models')

import torch
from model import Net


def test_model():
    batch_size = 1
    # Define an input to the model
    x = torch.randn((batch_size, 3, 256, 256))
    # Define the model
    model = Net()
    # Run the model
    output = model(x)
    # Check that output size is correct
    assert list(output.shape) == [batch_size, 11], 'Model output shape should be [batch_size, 11]'

test_model()