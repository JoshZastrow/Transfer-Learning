# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 18:54:35 2017

@author: Joshua Zastrow
"""


from data_utils import DataGenerator, DataWriter, stopwatch
import numpy as np 
from config import get_user_settings, create_parser
from keras import optimizers
from keras.layers import Dense, Dropout, MaxPooling2D
from keras.models import Sequential
import sys, traceback
import keras.applications

# If I go for random sampling... which I don't at the moment.
np.random.seed(7)
        
class FeatureGenerator():
    """
    Transfer Learning model.
    Convolutional Base Model Architecture, top layers (classifier) removed
    for low level feature genaration from raw data. Model pulls from a dataset
    and executes a prediction function to generate extracted features.
    
    args
    ----
        inputs (ndarray): 4D array of images (samples, height, width, channels)
        
    returns
    -------
        result (ndarray): generated features of input
    
    """
    def __init__(self, model, input_shape=(480, 640, 3)):
        conv_base = getattr(keras.applications, model)(
                include_top=False, 
                weights='imagenet', 
                input_shape=input_shape) 
        
        self.model = Sequential()
        self.model.add(conv_base)
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        
    def predict(self, inputs):
        return self.model.predict(inputs)
        

class RegressionModel():
    """
    A basic densely connected Neural Network for regression on extracted features
    from earlier feature extraction models.
    """
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(1024, 
                             activation='relu', 
                             input_shape=(5, 13, 18, 2048)))
        
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                           loss='mean_squared_error',
                           metrics=['accuracy'])
        
    def train(self, csv_file=None, img_dir=None):
        
        if not (csv_file or img_dir):
            csv_file = 'output/interpolated.csv'
            img_folder = 'output/'
            
        # No Longer functional -- needs to be replaced
        data_flow = DataGenerator()
        data_flow = data_flow.from_csv(csv_path=csv_file,
                                       img_dir=img_folder, 
                                       batch_size=5)
        
        return self.model.fit_generator(
               data_flow,
               samples_per_epoch=200,
               nb_epoch=5,
               verbose=2,
               nb_val_samples=100,
               validation_data=data_flow)
            
def module_argument_checker(args):
    model_options = ['InceptionV3', 'VGG16', 'Xception', 'ResNet50']
    assert args.model in model_options, 'input model is not an option'  
    

if __name__ == "__main__":
    import time
    start = time.time()
    
    # Get model settings from config file
    config = get_user_settings()
    get = config['USER']
    # get input informatin for data pipeline
    parser = create_parser()
    args = parser.parse_args()
    module_argument_checker(args)

    stopwatch(start, 'creating model and data generator')
    dfeed = DataGenerator(log_file=get['log file path'],
                          img_dir=get['image dir'],
                          batch_size=int(get['batch size']), 
                          sample_size=int(get['sample size']),
                          starting_row=int(get['starting row']))
   
    model = FeatureGenerator(args.model)
    store = DataWriter(args.output, args.model)
    i = 0
    
    stopwatch(start, 'running predictions')
    for imgs, labels in dfeed:            
        try:
            features = model.predict(imgs)
            store(features, labels)
            i += int(get['batch size'])
            stopwatch(start, 'data stored, next batch')     
            
        # If ctrl + c out of the data writer, store the row location
        except KeyboardInterrupt:
            print('keyboard interrupt, exiting program and saving row location')
            get['starting row'] = str(i)
            with open('config.ini', 'w') as configfile:
                config.write(configfile)
            sys.exit(0)
            
        except Exception:
            traceback.print_exc(file=sys.stdout)
            sys.exit(0)             

        # Save progress
        get['starting row'] = str(i)
        with open('config.ini', 'w') as configfile:
            config.write(configfile) 
