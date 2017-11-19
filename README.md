Title: Machine Learning - Transfer Learning Feature Extractor
Date: 2017-08-01 9:52
tags: Autonomous-Driving, Machine-Learning, Transfer-Learning
Category: Machine Learning
author: Josh Zastrow
summary: Getting Started with Comma-ai dataset

# Building a Transfer Learning Feature Extractor

I am working on a project to benchmark state-of-the-art machine learning models against a novel data processing algorithm. In order to compare the efficacy of data representation against multiple models, I will be employing Transfer Learning.

The following section will be a brief description of Transfer learning, feel free to skip to just get to running the model.

## What is Transfer Learning?

Transfer learning is the application of Machine Learning which takes the lower level outputs of a more sophisticated pre-trained model as a base model for a different application. 

There are several reasons why this is useful:

- The dataset on hand is small, and there is not enough available data to train a deep network from scratch. However the problem may require a more complex model.

- The problem is similar to a previous problem solved with a trained advanced model. Rather than spend time retraining an entire network, it can be useful to start  with a pre-trained network.

- Training a sophisticated network takes a considerable amount of time, money, and compute resources. Starting with a pre-trained network can shorten the development cycle considerably.

## Methods of Transfer Learning 

There are two ways to apply Transfer Learning techniques to a model:

- Transfer Learning as feature extraction, saving the output and using this transformed data as the new input.
- Transfer Learning as data processing, adding new untrained layers with random initialized values to the clipped base model and retraining.

### Feature Extraction Method

The first method is to clip the densely connected network segment from a pre-trained model, then running 'predictions' on the dataset to generate a new dataset. Since the classifier segment of the model has been removed, the resulting output would be low level representation of the raw data. This data can be saved and then used for further predictive modeling that deal with high level representation like classification or regression.

### Data Processing Method

The second method is to build a new model directly from a clipped base model. This  is where the pre-trained model would have it's densely connected layers removed, then new layers with randomized weights would be added to suit the context of the problem. The resulting model would be trained on the new dataset, either by freezing the weights of the base model (not training that segment) and only running backpropogation through the added layers, or if there is enough data on hand retraining the entire model which could improve the overall accuracy of the model. 

### Approach

For this project, I will be using the Feature Extraction method; saving the output of the base models as a new dataset.

## Setup

The following section covers getting the data, environment and model configurations setup to run the feature extractor.

    1. download the Udacity SDC Dataset
    2. copy the github repository
    3. create a virtual environment from YAML file
    4. configure model and data pipeline parameters

### Dataset
The data used for this project was Udacity's Self Driving Car Dataset. Before continuing you should have the data extracted from the downloaded ROSbags into the respective output folders. 

If you do not have the Udacity dataset, see this article on how to obtain it:

    https://github.com/udacity/self-driving-car
    
    
Note: if you are interested in using a different dataset for transfer learning that is fine; however the data generator function of this program is based off Udacity's data structure; a log file that has a column referencing the path to an image file, and an image folder.

See **insert github repo link** for the data generator function

### Github Repository

clone the repository to the same directory.

    hyperlink HERE
    
    
### Environment

The repository contains an environment yaml file that can be used for setting up a python environment for running the code. This will ensure that you have the same dependencies as what was used to run the model. 

I have the Anaconda distribution and conda package manager, so I setup my python environments using Conda. If you have anaconda installed, open the terminal/command prompt, navigate to the local repo directory and enter:

    conda create -n transferlearning --file=environment.yml

Alternatively, outside the scope of this article but you may be able to create an environment from file using virtualenv, or you could skip the environment and install the packages listed in the file. 


Not all the libraries in the environment file were used in the final code, alternatively you could read through the imported library list in model.py, config.py and data_utils.py and make sure you have those libraries installed.

### Model Configuration

Before running the model, you should configure the model parameters as well as the settings for the data pipeline. This includes setting the batch size, the number of samples to run, the file path to the log, the image directory folder

All the pertinent files are in the Transfer-Learning folder, so navigate there before running any programs:

    cd Transfer-Learning

Run the config program from terminal/cmd:

    python config.py
    
This will prompt for a few integer based parameters, all of which are optional. You may want to set the sample size though, because the default is just 20 images. I would suggest for the first run to just set the sample size and leave the rest blank.  

# Feature Extraction

To perform feature extraction on your dataset, run the model.py program. The [model]() is actually quite simple due to the tools offered with Keras. 

My recommendation is to first run the helper on the program to get an outline of the required arguments for the program:

    python model.py --help
    
The program needs to know the output folder destination and the type of model to use for feature extraction. To execute the code the command line prompt would look something like this:

    python model.py   \
    -model "InceptionV3" \
    -output "output/results"
    
Running the program will process images from your dataset through the base model and output the results to an output folder.

If you would like to change how or where the resulting arrays are stored outside the settings of the config, the code for the data writer portion can be found [here](). 

# Results

For the purpose of demonstration, I will feed one example through each model. The resulting data transformations is a collection of sparse activations far removed from the data rich image input. 


The code for this visualizer can be found [here]().
The abstraction done by these models is a bit of a black box, so the raw output of these transformations are difficult to interpolate at a glance.

The original image is essentially a pixel window of several hundred values both in height and width, with three channels/filters of color. The resulting transformation leaves us with 2048 filters (512 for VGG) on very small pixel window--only a couple across.

The next steps with this newly processed data is to built a densely connected classifier to connect these inputs to ground truth labels. This secondary model will perform high level classification/regression to get use-able results. Happy coding!


        
        