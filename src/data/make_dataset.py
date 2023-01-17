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
    convert_totensor = transforms.ToTensor()

    n_imgs_train = 0
    for dirs in train_dirs[1:]: 
        n_imgs_train += len(os.listdir(dirs))

    n_imgs_valid = 0
    for dirs in valid_dirs[1:]: 
        n_imgs_valid += len(os.listdir(dirs))

    train_images = []
    train_labels = []

    c = 0
    for idx, dir in enumerate(train_dirs[1:]):
        images_dir = os.listdir(dir)
        label=dir.split("\\")[-1]
        label_id = idx
        for image_dir in images_dir:
            try:
                with Image.open(dir+'/'+image_dir) as img:
                    # Check if file is an image
                    if img.format in ['JPEG', 'PNG', 'GIF', 'BMP', 'TIFF']:
                        img.verify()

                        image_original = (Image.open(dir+'/'+image_dir))
                        
                        # If image has 4 channels, convert it to RGB
                        if image_original.mode == 'RGBA':
                            image_original = image_original.convert('RGB')
                        # Resize image
                        if image_original.size != (256, 256):
                            image_resized = image_original.resize((256, 256))
                        else: 
                            image_resized = image_original
                        # Convert image to tensor
                        image = convert_totensor(image_resized)
                        # Append images in a list
                        train_images.append(image)
                        train_labels.append(label_id)
                        c += 1
                        print(f"Images: {c}/{n_imgs_train}")
                    else:
                        os.remove(dir+'/'+image_dir)
                        print(f'{dir}/{image_dir} was removed')
            except (IOError, SyntaxError) as e:
                os.remove(dir+'/'+image_dir)
                print(f'{dir}/{image_dir} was removed')

    # Save train images and train labels
    torch.save(torch.stack(train_images), output_filepath + '/train_images.pt')
    torch.save(torch.tensor(train_labels), output_filepath + '/train_labels.pt')

    valid_images = []
    valid_labels = []

    c = 0
    for idx, dir in enumerate(valid_dirs[1:]):
        images_dir = os.listdir(dir)
        label=dir.split("\\")[-1]
        label_id = idx
        for image_dir in images_dir:
            try:
                with Image.open(dir+'/'+image_dir) as img:
                    # Check if file is an image
                    if img.format in ['JPEG', 'PNG', 'GIF', 'BMP', 'TIFF']:
                        img.verify()
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
                        
                        valid_images.append(image)
                        valid_labels.append(label_id)
                        c += 1
                        print(f"Images: {c}/{n_imgs_valid}")
                    else:
                        os.remove(dir+'/'+image_dir)
                        print(f'{dir}/{image_dir} was removed')
            except (IOError, SyntaxError) as e:
                os.remove(dir+'/'+image_dir)
                print(f'{dir}/{image_dir} was removed')

    torch.save(torch.stack(valid_images), output_filepath + '/valid_images.pt')
    torch.save(torch.tensor(valid_labels), output_filepath + '/valid_labels.pt')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
 
    main("/data/raw/tomato-disease-multiple-sources", "./data/processed")  #input_filepath, output_filepath