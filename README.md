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

The second method is to build a new model directly from a clipped base model. This  is where the pre-trained model would have it's densely connected layers removed, then new layers with randomized weights would be added to suit the context of the problem. Usually the new layers are just one or two standard affine layers connected to the output, but in the case of autonomous driving, sometimes it can be an RNN to recall recently seen examples.

The resulting model would be trained on the new dataset, either by freezing the weights of the base model (not training that segment) and only running backpropogation through the added layers, or retraining the entire model. If retrained, the further refinement of the weights on the pretrained network could improve the overall accuracy of the model. 

### Approach

For this project, I will be using the Feature Extraction method; saving the output of the base models as a new dataset.

## Setup

The following section covers getting the data, environment and model configurations setup to run the feature extractor.

    1. download the Udacity SDC Dataset
    2. copy the github repository
    3. create a virtual environment from YAML file
    4. configure model and data pipeline parameters

### Dataset

The data used for this project was Udacity's Self Driving Car Dataset. I included a small segment of this dataset (3000 images) in the transfer learning repository so that the code can be run right out of the box.

If you want access to the full dataset, navigate to Udacity's dataset github page on how to obtain it:

    https://github.com/udacity/self-driving-car
    
 **Note:** if you are interested in using a different dataset for transfer learning that is fine; however you may need to modify the data generator function or organize your data to match Udacity's data structure: 
 
    +-- data/
    |   +-- log file <.csv>
    |   +-- left/
    |   |   +-- image #1 <.jpg>
    |   |   +-- image #2 <.jpg>       
    |   |   +-- ....     <.jpg>
    |   +-- center/
    |   |   +-- image #1 <.jpg>
    |   |   +-- image #2 <.jpg>       
    |   |   +-- ....     <.jpg>
    |   +-- right/
    |   |   +-- image #1 <.jpg>
    |   |   +-- image #2 <.jpg>       
    |   |   +-- ....     <.jpg>
See [data_utils.py](https://github.com/JoshZastrow/Transfer-Learning/blob/master/data_utils.py#L16) for the data generator function. 

### Github Repository

Clone the repository. There is a "data/" folder that holds the sample data. If you are using your own dataset then you could move your data into that folder.

    https://github.com/JoshZastrow/Transfer-Learning
    
    
### Environment

The repository contains an environment yaml file that can be used for setting up a python environment for running the code. This will ensure that you have the same dependencies as what was used to run the model. 

I have the Anaconda distribution and conda package manager, so I setup my python environments using Conda. If you have anaconda installed, open the terminal/command prompt, navigate to the local repo directory and enter:

    conda create -n transfer-learning --file=environment.yml

Alternatively, outside the scope of this article but you may be able to create an environment from file using virtualenv, or you could skip the environment and install the packages listed in the file. 


Not all the libraries in the environment file were used in the final code, so you could read through the imported libraries in these files:
* model.py
* config.py
* data_utils.py  
* visualizer.py 

and make sure you have those libraries installed.

### Model Configuration

Before running the model, you should configure the model parameters as well as the settings for the data pipeline. This includes setting the batch size, the number of samples to run, the file path to the log, the image directory folder

All the pertinent files are in the Transfer-Learning folder, so navigate there before running any programs:

    cd Transfer-Learning

Run the config program from terminal/cmd:

    python config.py
    
This will prompt for a few integer based parameters, all of which are optional. You may want to set the sample size though, because the default is just 20 images. I would suggest for the first run to just set the sample size and leave the rest blank.  

# Feature Extraction

To perform feature extraction on your dataset, run the model.py program. The [model](https://github.com/JoshZastrow/Transfer-Learning/blob/master/model.py#L13) is actually quite simple due to the tools offered with Keras. 

My recommendation is to first run the helper on the program to get an outline of the required arguments for the program:

    python model.py --help
    
The program needs to know the output folder destination and the type of model to use for feature extraction. To execute the code the command line prompt would look something like this:

    python model.py   \
    -model "InceptionV3" \
    -output "data"
    
Running the program will process images from your dataset through the base model and output the results to an output folder.

If you would like to change how or where the resulting arrays are stored outside the settings of the config, the code for the data writer portion can be found [here](https://github.com/JoshZastrow/Transfer-Learning/blob/master/data_utils.py#L103). 

# Results

For the purpose of demonstration, I will feed one example through each model. The resulting data transformations is a collection of sparse activations far removed from the data rich image input. 

![Transfer Model's interpretation of a cat](https://github.com/JoshZastrow/Transfer-Learning/blob/master/results/CatResults.png)

The code for this visualizer can be found [here](https://github.com/JoshZastrow/Transfer-Learning/blob/master/visualizer.py#L23). It also shows how to make use of the model without the accompanying data pipeline.

The abstraction done by these models is a bit of a black box, so the raw output of these transformations are difficult to interpolate at a glance.

The original image is essentially a pixel window of several hundred values both in height and width, with only three channels/filters of color. The resulting transformation leaves us a very small pixel window--only a couple pixels across, but with many filters/channels. The resulting 2048 filters were organized in 32 x 64 tiles.

# Next Steps

The next steps with this newly processed data is to build and train a densely connected classifier or regression model on these new inputs in relation to ground truth labels. Once trained, this secondary model will perform high level classification/regression to get use-able results. 

For the data pipeline, it would be useful to have additional pre-processing functions before feeding the raw images to the base models. There is an [image processing](<https://github.com/JoshZastrow/Transfer-Learning/blob/master/data_utils.py#L70>) function that allows for transformation of images once read from file.

Happy coding!


        
        