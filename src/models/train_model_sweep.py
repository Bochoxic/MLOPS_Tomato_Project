import torch
import wandb
import yaml
import sys
import os

from train_functions import train, train_lightning

def main():
    project_path=os.getcwd()
    print("Training with sweep")

    with open(project_path + '/config/wandb_sweep.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    wandb.init(config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using torch device of type {device.type}{": " + torch.cuda.get_device_name(device) if device.type == "cuda" else ""}')
    
    print("Start Training...")
    # Extract values from config file

    model_type = wandb.config.model_type
    lr = wandb.config.lr
    batch_size = wandb.config.batch_size
    n_epoch = wandb.config.epochs
    image_size_x = wandb.config.image_size_x
    image_size_y = wandb.config.image_size_y
    limit_batches = wandb.config.limit_batches
    patience = wandb.config.patience


    if wandb.config.training_type == 'lightning': 
        train_lightning(project_path, image_size_x, image_size_y, model_type, batch_size, lr, n_epoch, limit_batches, patience)
    elif wandb.config.training_type == 'classic':   
        train(project_path, image_size_x, image_size_y, device, model_type, batch_size, lr, n_epoch)
    else:
        sys.exit("Training is not defined correctly in hydra file")

main()