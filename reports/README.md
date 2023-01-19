# MLOPS_Tomato_Project

Vicente Bosch & Paloma Guiral & Víctor Vigara

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


---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [ ] Construct one or multiple docker files for your code
* [ ] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [x] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [ ] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training in GCP using either the Engine or Vertex AI
* [ ] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

45

### Question 2
> **Enter the study number for each member in the group**

> Answer:

s222928, s221924, s222929

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Answer:

We are using the PyTorch Image Models framework, because our project is about the classification of tomato diseases from tomato leaves images
Among all the models offered by this framework, we have decided to choose 'resnet' as it is intended for image classification.
In addition, when loading the model, this framework allows us to select a pre-trained model from which to start as a basis for better training.
After training the pre-trained model with our images, an optimal classification was obtained.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

--- We used conda to create a virtual environment to avoid dependencies version issue between different projects. However, we have used pip to install the dependencies in the conda environment. 
The list of dependencies was auto-generated using the package 'pipreqs', that automatically scan your project and create a requirement file with all the packages that you import in your code. Every time we want to update the requirement file we have to run the following comand: pipreq --force . 'force' argument is to overwrite the old file.
A new member should clone the github project repository and execute the following command: pip install -r requirements.txt. With that command, all the dependencies generated previously are installed.---

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> 
> Answer:
cookie-cutter is standardized way of creating project structures, making one able to faster get understand the other persons code. It just a template and maybe not all parts of it are important for our project. From the cookiecutter template we have filled out the 'src' folder:
- src: here we have filled out the following folders: 
      
      -'data' where is the code to download and make the datasets for our model. 
      - 'models' where is the code to create the network, train and predict.
We have added the following folders for running our experiments: 
- ./dvc: contains a pointer to your remote storage 
- .github: .github/workflows/ contains different workflows
- config: contains config files to keep track of hyperparameters
- reports: contains the project description and exam
-tests: contains the unit testing to tests individual parts of your code base.


### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

A part from the structure, to maintain the quality and format of the code we have implemented:
-     Documentation: to remember all details about the code.
-	Styling: we used flake8 to check if our code is pep8 (the official style guide for python). Important when working multiple people together who have different coding styles. We also have taken care of the import statements, applying the isort standard.
-	Typing: specify data types for variables. You can know the expected types of input arguments and returns by just reading the code.


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total we have implemented 3 tests, where we test our data, the model and the training process.
-test_data: checks for both the test data and the test train, that the shape of each image is [batch_size, 3, 256, 256] and the labels size should be like the batch size
-test_model: applies the model to an input, and checks that the output size is correct ([batch_size, 11])
-test_training: trains the model and checks that: the training loss is >=0, validation loss is >=0 and validation accuracy is <=1


### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- question 8 fill here ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

--- question 9 fill here ---

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did make use of DVC in the following way: Firstly, we made dvc work together with our own Google drive to storage data. However, a big limitation of this is that we need to authentic each time we try to either push or pull the data. Therefore, we need to use an API instead which is offered through gcp. So, we created a bucket through the GCP page and after that, we changed the storage from our Google drive to this new Google cloud storage and pushed the data to the cloud.

Having a version control of our data has helped us in the development of our project in several weeks: has made it easy to understand how the data has evolved; has allow the three members of the team to work on the data simultaneously without conflicts or data loss; it has been easy to reproduce the exact state of the data at any point in the project; it has provided safety net in case of data loss or corruption; allows to track different versions of the data and the corresponding results to choose the best result 

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

We have made use of: Unittesting, Github actions and Lightning.
-	Unittesting. Test that tests individual parts of the code base to test for correctness. We created the folder ‘tests’, where we have implemented three different tests:
      o     tets_data.py: checks that the shape of each image is [batch_size, 3, 256, 256] and the labels size should be like the batch size. Also checks if the data is present at /data.
      o     test_model.py: checks that the output size, after applying the model, is correct ([batch_size, 11]). 
      o     test_training.py: trains the model and checks that: the training loss is >=0, validation loss is >=0 and validation accuracy is <=1
-	Github actions. To automatize the testing, such that it done every time we push to our repository.
We store our different workflows at the folder .github/workflows:
      o	tests.yml: run the tests for us
      o	isort.yml: runs isort on the repository
      o	flake8.yml: runs flake8 on the repository
-	Pytorch Lightning. Adding the LightningModule to the first approach of our model.py, and two new methods needed: ‘training_step’ and ‘configure_optimizers’
                                                                                                                                    
We don't make use of caching.                                                                                                        **EXAMPLE OF A TRIGGERED WORKFLOW --> FALTA**
 ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We use Hydra, a tool to write config file to keep track of hyperparameters, with the structure:
`|--config
|  |--default_config.yaml
|  |--experiment
|   |--exp1.yaml`
            
‘default_config.yaml’ points to the experiment that we want to run. That experiment is contained in the folder ‘experiment’ with the hyperparameters needed to run the script (‘batch_size’, ‘lr’, ‘n_epoch’, ‘limit_batches’ and if we want to run the training with the lightning api or without it) and a value.
We load the configuration file inside our script using hydra, and to run our training calling the train_model.py from the terminal:
                                `python src/models/train_model.py`
            
            
### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We made use of config files , applying Hydra, a configuration tool that is based around writing those config files to keep track of hyperparameters and save them with the experiment.
Whenever an experiment is run the following happens:
-	we have to specify the hyperparameters values in a .yaml file in config/experiment 
-	load the configuration file inside your script (using hydra) that incorporates the hyperparameters into the script
-	Run the script
-	By default hydra will write the results to a ‘outputs’ folder
To reproduce an experiment one would have to choose the .yaml file of the experiment wanted and run the script providing that configuration file as an argument. In our case, right now, we just have one experiment (‘exp1.yaml), so we run the training script without argument: `python src/models/train_model.py`


### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

We run the different scripts in VS, so when running into bugs we inserted inline breakpoints in the code and then execute the script in debug mode. 
The code runs until the breakpoint, and after that we can run the rest of the code line by line, so we can see at which point of the code the model is failing and we can also see the value of the different variables using the debug console.
                                          
                                          **PROFILE?**


## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used the following services:
-	Compute Engine: to create and run a virtual machine, which has allowed us to essentially run an operating system that behaves like a completely separate computer. After creating an appropriate VM we log into it and run our code in that machine. **COMENTAR CARACTERÍSTICAS ETC**
-	Cloud storage: to store the data in the cloud to make it easier to share, expand and not to lose it.
-	Container registry: **COMPLETAR**


### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:
![question19](https://user-images.githubusercontent.com/99659050/213477437-9e3acc6e-0f42-498b-916b-8bb3a3d006a4.jpeg)

            
### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:
![question 20](https://user-images.githubusercontent.com/99659050/213477525-03772243-08d4-49a1-b67f-7c5e9cffc493.jpeg)


### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:
![question 21](https://user-images.githubusercontent.com/99659050/213477589-37df1549-079a-468f-b892-babb582b027b.jpeg)


### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 22 fill here ---

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- question 24 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 26 fill here ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 27 fill here ---

