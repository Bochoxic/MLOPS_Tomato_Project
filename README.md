# MLOPS_Tomato_Project

Vicente Bosch
Paloma Guiral
Víctor Vigara

## Overall goal of the project
The goal of the project is to use image models to predict and classify wich type of disease has a tomato leaf. 

## What framework are you going to use (PyTorch Image Models, Transformes and PyTorch-Geometric)
Our initial intention is to use the PyTorch Image Models framework. 

## How do you intend to include the framework into your project
We want to use a pre-trained model, there are a lot of them, so we will study and choose the one which fits better to our project.

## What data are you going to run on
We are using the Kaggle [Tomato Disease Multiple Sources](https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources?datasetId=2516350&sortBy=voteCount) Dataset. Over 20k images of tomato leaves with 10 diseases and 1 healthy class. Images are collected from both lab scenes and in-the-wild scenes.

Classes: 
- Late_blight
- healthy
- Early_blight
- Septorialeafspot
- TomatoYellowLeafCurlVirus
- Bacterial_spot
- Target_Spot
- Tomatomosaicvirus
- Leaf_Mold
- Spidermites Two-spottedspider_mite
- Powdery Mildew

The original source of most of the images is the PlantVillage dataset published here and here. The data has been augmented offline using multiple advanced techniques like image flipping, Gamma correction, noise injection, PCA color augmentation, rotation, and scaling. Some recent images were generated offline with GANs. The subset of images containing Taiwan tomato leaves was augmented using rotations at multiple angles, mirroring, reducing image brightness, etc.

## What deep learning models do you expect to use
We are still determining which models we will train, but previous works suggest that Convolutional and Resnet networks fit well with the dataset.
