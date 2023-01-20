import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import os
import torch.nn.functional as F
from model import Net
import torch

def accuracy():

    image_size_x, image_size_y = 256, 256
    batch_size = 64
    project_path = os.getcwd()

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

    # Load model
    model = Net(0.0001)
    model.eval()
    model.load_state_dict(torch.load('models/lightning/trained_model.pt'))


    images, labels = next(iter(testloader))
    # Get the class probabilities
    ps = F.softmax(model(images))
    # Most likely classes 
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    # Get model accuracy
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f'Accuracy: {accuracy.item()*100}%')

accuracy()