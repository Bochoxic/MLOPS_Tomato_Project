import hydra
import wandb
import torch
import sys
import os

from train_functions import train, train_lightning

project_path=os.getcwd()

# Load config file with hydra
@hydra.main(config_path="../../config", config_name='default_config.yaml') #config/default_config.yaml
def main(config):  
    
    print("Training without sweep")
    wandb.init(project="Tomato Project")
    # Use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using torch device of type {device.type}{": " + torch.cuda.get_device_name(device) if device.type == "cuda" else ""}')
    print("Start Training...")

    model_type = config.experiment.model
    hyperparams = config.experiment.hyperparams
    image_size_x = config.experiment.image_size_x
    image_size_y = config.experiment.image_size_y
    patience = config.experiment.patience

    lr = hyperparams["lr"]
    batch_size = hyperparams["batch_size"]
    n_epoch = hyperparams["n_epoch"]
    limit_batches = hyperparams["limit_batches"]

    # Select between normal or lightning training (config/experiment/exp1/training: lightning or normal)
    if config.experiment.training == 'lightning': 
        train_lightning(project_path, image_size_x, image_size_y, model_type, batch_size, lr, n_epoch, limit_batches, patience)
    elif config.experiment.training == 'classic':   
        train(project_path, image_size_x, image_size_y, device, model_type, batch_size, lr, n_epoch)
    else:
        sys.exit("Training is not defined correctly in hydra file")

main()