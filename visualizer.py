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
    global VGG16
    
    layers = [layer.output for layer in VGG16.layers]
    
    # Create a multiple output model
    # Each layer is an output in this model.
    activations = models.Model(input=VGG16.input,
                               output=layers).predict(img)
    
    # Animate the transformations
    fig = plt.figure(facecolor='grey')
    gs1 = gridspec.GridSpec(1, 4)
    gs2 = gridspec.GridSpec(1, 4)
    gs1.update(left=0.05, right=0.95, wspace=0.05, top=0.8, bottom=0.4)
    gs2.update(left=0.05, right=0.95, wspace=0.05, top=0.39, bottom=0.25)
    
    img_ax = []
    act_ax = []
    images = []
    labels = []
    nodes = []  
    
    # Create subplots, images and labels
    for i in range(4):
        # Add a subplot for each image
        img_ax += [fig.add_subplot(gs1[0, i])]
        act_ax += [fig.add_subplot(gs2[0, i])]
        # remove tick marks
        img_ax[i].set_xticks([])
        img_ax[i].set_yticks([])
        act_ax[i].set_xticks([])
        act_ax[i].set_yticks([])  
        
        # put an image and label in each subplot
        images += [img_ax[i].imshow(activations[0][0], animated=True)]
        labels += [img_ax[i].text(.50, 0.90, '', fontsize=14,
                                  horizontalalignment='center',
                                  transform=img_ax[i].transAxes, 
                                  color='white')]
  
        # add a filter activation images
        mask = np.zeros(shape=(1, activations[0].shape[3]))
        nodes += [act_ax[i].imshow(mask)]
    
    def init():
        for img in images:
            img.set_data(activations[0][0])
        
        for txt in labels:
            txt.set_text('')
        
        for idx in nodes:
            idx.set_data(mask)
        
        return images + labels + nodes

    
    def layer_generator():
        VGG16_layers = zip(VGG16.layers, activations)
        
        for layer, output in VGG16_layers:
            name = layer.name
            output_filters = output.shape[3]

            for i in range(output_filters):
                filter_image = output[0, :, : , i]
                filter_mask = np.zeros(shape=(1, output_filters))
                
                if output_filters % 64 == 0:
                    filter_mask = filter_mask.reshape((-1, 64))
                
                filter_mask.itemset(i, 1)
                yield [name, filter_image, filter_mask]
                          
                
    def plotter(args):
        name, output, mask = args
        
        for img in images:
            img.set_data(output)
        for txt in labels:
            txt.set_text(name)
        for i in range(4):
            nodes[i] = act_ax[i].imshow(mask)
            
        res = images + labels + nodes
        
        return res
        
    anim = animation.FuncAnimation(fig, plotter,  init_func=init,
                                   frames=layer_generator,
                                   interval=5,
                                   blit=True,
                                   repeat=False)
    
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
    
    
    