# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:34:35 2017
Code used to generate output visualization
@author: joshua
"""

import imageio as io
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import animation
from model import FeatureGenerator
from keras import applications
from keras import models

def visualize_layers():
    img = io.imread('britishBlue.jpg')
    img = img / 255
    img = np.expand_dims(img, axis=0)
    
    VGG16 = applications.VGG16(
                include_top=False, 
                weights='imagenet', 
                input_shape=img[0].shape) 
    
    layers = [layer.output for layer in VGG16.layers[:4]]
    
    # Create a multiple output model
    # Each layer is an output in this model.
    activations = models.Model(input=VGG16.input,
                               output=layers).predict(img)
    
    # Create image list
    layer_outputs = []
    for layer in activations:
        for filter in range(layer.shape[3]):
            layer_outputs.append(layer[0, :, :, filter])
            
    # Animate the transformations
    fig = plt.figure()
    im = plt.imshow(layer_outputs.pop(0))
    
    def init():
        return [im]
    
    def plotter(x):
        im = plt.imshow(x, animate=True),
        return [im]
    
    anim = animation.FuncAnimation(fig, plotter, 
                                   init_func=init,
                                   frames=layer_outputs[:20],
                                   interval=20,
                                   blit=True)
    
    plt.show()
    
    
def visualize_outputs():
            
    # Get output
    img = io.imread('britishBlue.jpg')
    img = np.expand_dims(img, 0)
    
    print('Input shape:', img[0].shape)
    print('Output shapes:')
    
    # Transform data
    for model in ['InceptionV3', 'ResNet50', 'Xception', 'VGG16']:
        output = FeatureGenerator(model, input_shape=img[0].shape).predict(img)
        output = output[0]
    
        print('\t{}: {}'.format(model, output.shape))
    
        # Visualize results
        if model == 'VGG16':
            height, width = 16, 32
        else:
            height, width = 32, 64
    
        row, col, _ = output.shape
        output = output.reshape(row, col, height, width)
    
        fig = plt.figure(figsize=(6, 3), dpi=100)
        fig.suptitle(model)
        gs = gridspec.GridSpec(row, col, wspace=0.1, hspace=0.1)
    
        for i in range(row):
            for j in range(col):
                axs = plt.subplot(gs[i, j])
                axs.imshow(output[i, j, :, :], cmap='gray')
                axs.axis('off')
    
        fig.savefig('results/{}.png'.format(model))
   

if __name__ == '__main__':
    visualize_layers()   