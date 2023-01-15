# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import opendatasets as od
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import ImageFile

#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    project_dir = os.getcwd()
    #train
    train_dirs = [x[0] for x in os.walk(project_dir+input_filepath+"/train")]
    valid_dirs = [x[0] for x in os.walk(project_dir+input_filepath+"/valid")]

    n_imgs_train = 0
    for dirs in train_dirs[1:]: 
        n_imgs_train += len(os.listdir(dirs))

    n_imgs_valid = 0
    for dirs in valid_dirs[1:]: 
        n_imgs_valid += len(os.listdir(dirs))

    train_images = torch.empty((n_imgs_train, 3, 256, 256))
    train_labels = torch.empty(n_imgs_train)

    convert_totensor = transforms.ToTensor()
    c = 0
    for idx, dir in enumerate(train_dirs[1:]):
        images_dir = os.listdir(dir)
        label=dir.split("\\")[-1]
        label_id = idx
        for image_dir in images_dir:
            
            image_original = (Image.open(dir+'/'+image_dir))

            if image_original.mode == 'RGBA':
                image_original = image_original.convert('RGB')

            if image_original.size != (256, 256):
                image_resized = image_original.resize((256, 256))
            else: 
                image_resized = image_original
            
            image = convert_totensor(image_resized)
            
            try: 
                train_images[c] = image
            except: 
                a=1
            train_labels[c] = label_id
            c += 1
            print(f"Images: {c}/{n_imgs_train}")
            

    torch.save(train_images, output_filepath + '/train_images.pt')
    torch.save(train_labels, output_filepath + '/train_labels.pt')

    valid_images = torch.empty((n_imgs_valid, 3, 256, 256))
    valid_labels = torch.empty(n_imgs_valid)

    c = 0
    for idx, dir in enumerate(valid_dirs[1:]):
        images_dir = os.listdir(dir)
        label=dir.split("\\")[-1]
        label_id = idx
        for image_dir in images_dir:
            
            image_original = (Image.open(dir+'/'+image_dir))

            if image_original.mode == 'RGBA':
                try: 
                    image_original = image_original.convert('RGB')
                except: 
                    a = 1
            if image_original.size != (256, 256):
                image_resized = image_original.resize((256, 256))
            else: 
                image_resized = image_original
            
            image = convert_totensor(image_resized)
            
            valid_images[c] = image
            valid_labels[c] = label_id
            c += 1
            print(f"Images: {c}/{n_imgs_valid}")


    torch.save(valid_images, output_filepath + '/valid_images.pt')
    torch.save(valid_labels, output_filepath + '/valid_labels.pt')



    

    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
 
    main("/data/raw/tomato-disease-multiple-sources", "./data/processed")  #input_filepath, output_filepath