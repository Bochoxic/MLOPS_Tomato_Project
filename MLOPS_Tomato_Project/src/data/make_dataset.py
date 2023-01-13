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


#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    project_dir = os.getcwd()

    #train
    train_dirs = [x[0] for x in os.walk(project_dir+input_filepath+"/train")]

    train_images = []
    train_labels = []
    convert_totensor = transforms.ToTensor()
    for dir in train_dirs[1:]:
        images_dir = os.listdir(dir)
        label=dir.split("\\")[-1]
        for image_dir in images_dir:
            image = convert_totensor((Image.open(dir+'/'+image_dir)))
            train_images.append(image)
            train_labels.append(label)

    #Valid
    valid_dirs = [x[0] for x in os.walk(project_dir+input_filepath+"/valid")]
    valid_images = []
    valid_labels = []
    convert_totensor = transforms.ToTensor()
    for dir in valid_dirs[1:]:
        images_dir = os.listdir(dir)
        label=dir.split("\\")[-1]
        for image_dir in images_dir:
            image = convert_totensor((Image.open(dir+'/'+image_dir)))
            valid_images.append(image)
            valid_labels.append(label)
            

    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main("/data/raw/tomato-disease-multiple-sources", "jgj")
