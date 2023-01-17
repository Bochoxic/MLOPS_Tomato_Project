import os

import torch
from torch.utils.data import Dataset


class tomatoDataset(Dataset): 
    def __init__(self, data_type, data_dir):
        super().__init__()

        self.data_type = data_type
        self.data_dir = data_dir
        self.images, self.labels = self._load_data()

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
        
    def _load_data(self): 
        project_dir = os.getcwd()
        images_dir =  self.data_dir+"/"+self.data_type+"_images.pt"
        labels_dir =  self.data_dir+"/"+self.data_type+"_labels.pt"
        images = torch.load(images_dir)
        labels = torch.load(labels_dir)

        return images, labels