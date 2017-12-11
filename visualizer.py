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
    
    # Load in image, convert to tensor shape
    img = io.imread('britishBlue.jpg')
    img = img / 255
    img = np.expand_dims(img, axis=0)
    
    
    architectures = ['VGG16', 'InceptionV3', 'Xception', 'ResNet50']

    global deepNN
    
    deepNN = {}
    deepNN['model'] = {}
    deepNN['layers'] = {}
    deepNN['activations'] = {}
    deepNN['count'] = len(architectures)
    
     # Animate the transformations
    fig = plt.figure(facecolor='grey')
    gs1 = gridspec.GridSpec(1, deepNN['count'])
    gs2 = gridspec.GridSpec(1, deepNN['count'])
    
    gs1.update(left=0.05, right=0.95, wspace=0.05, top=0.8, bottom=0.4)
    gs2.update(left=0.05, right=0.95, wspace=0.05, top=0.39, bottom=0.25)
    
    img_ax = []
    act_ax = []
    images = []
    labels = []
    nodes = []
    i = 0
    
    for name in architectures:
        
        # Create Model and activations
        deepNN['model'][name] = getattr(applications, name)(
                            include_top=False, 
                            weights='imagenet', 
                            input_shape=img[0].shape) 
        
        layer_outputs = [layer.output for layer in deepNN['model'][name].layers]
        
        deepNN['activations'][name] = models.Model(input=deepNN['model'][name].input, 
                                                  output=layer_outputs).predict(img)
        
        # add subplots
        print('adding subplot ',i)
        img_ax += [fig.add_subplot(gs1[0, i])]
        act_ax += [fig.add_subplot(gs2[0, i])]
        
        # remove tick marks
        img_ax[i].set_xticks([])
        img_ax[i].set_yticks([])
        act_ax[i].set_xticks([])
        act_ax[i].set_yticks([])  
        img_ax[i].set_title(name)
        act_ax[i].set_title('Filter Activation')
        # put an image and label in each subplot
        images += [img_ax[i].imshow(deepNN['activations'][name][0][0], animated=True)]
        labels += [img_ax[i].text(.50, 0.90, '', fontsize=14,
                                  horizontalalignment='center',
                                  transform=img_ax[i].transAxes, 
                                  color='white')]
  
        # add a filter activation images
        mask = np.zeros(shape=(1, deepNN['activations'][name][0].shape[3]))
        nodes += [act_ax[i].imshow(mask)]
        
        i += 1


    def layer_generator():
        
        # architectures
        architectures = deepNN['model'].keys()
        # Initialize dictionary of frame lists
        frame_data = {model_name: [] for model_name in architectures}
        model_count = len(deepNN.keys())
        
        # keep track of how many frames there are
        frame_data['count'] = {}
        frame_data['max'] = 0
        
        for name in architectures:
            layer_names = [l.name for l in deepNN['model'][name].layers]
            
            for ID, output in zip(layer_names, deepNN['activations'][name]):
                filter_count = output.shape[3]
        
                for x in range(filter_count):
                    
                    # image visualization
                    filter_image = output[0, :, : , x]
                    
                    # filter visualization
                    filter_mask = np.zeros(shape=(1, filter_count))
                     
                    if filter_count % 64 == 0:
                        filter_mask = filter_mask.reshape((-1, 64))
                    
                    # set index of visualized filter
                    filter_mask.itemset(x, 1)
                    
                    data = {'name':ID, 
                            'image': filter_image,
                            'filter': filter_mask, 
                            'model_count': model_count}
                    
                    frame_data[name].append(data)

            
            # Figure out what frames to incrememt by
            frame_data['count'][name] = len(frame_data[name])
            
            if frame_data['max'] < frame_data['count'][name]:
                frame_data['max'] = frame_data['count'][name]
                
        while True:
            data_packet = {}
            
            for f in range(frame_data['max']):
                for name in architectures:
                    idx =  f % frame_data['count'][name]
                    data_packet[name] = frame_data[name][idx]
                
                yield data_packet


    def plotter(args):
        k = 0
        for name, data_packet in args.items():
            images[k].set_data(data_packet['image'])
            labels[k].set_text('Layer ' + data_packet['name'])
            nodes[k] = act_ax[k].imshow(data_packet['filter'])
            k += 1
        res = images + labels + nodes

        return res
        
    anim = animation.FuncAnimation(fig, plotter,
                                   frames=layer_generator,
                                   interval=1,
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
            height, wnameth = 16, 32
        else:
            height, wnameth = 32, 64
    
        row, col, _ = output.shape
        output = output.reshape(row, col, height, wnameth)
    
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
    
    # Load in image, convert to tensor shape
    img = io.imread('britishBlue.jpg')
    img = img / 255
    img = np.expand_dims(img, axis=0)
    
    
    architectures = ['VGG16', 'InceptionV3', 'Xception', 'ResNet50']

    global deepNN
    
    deepNN = {}
    deepNN['model'] = {}
    deepNN['layers'] = {}
    deepNN['activations'] = {}
    deepNN['count'] = len(architectures)
    
     # Animate the transformations
    fig = plt.figure(facecolor='grey')
    gs1 = gridspec.GridSpec(1, deepNN['count'])
    gs2 = gridspec.GridSpec(1, deepNN['count'])
    
    gs1.update(left=0.05, right=0.95, wspace=0.05, top=0.8, bottom=0.4)
    gs2.update(left=0.05, right=0.95, wspace=0.05, top=0.39, bottom=0.25)
    
    img_ax = []
    act_ax = []
    images = []
    labels = []
    nodes = []
    i = 0
    
    for name in architectures:
        
        # Create Model and activations
        deepNN['model'][name] = getattr(applications, name)(
                            include_top=False, 
                            weights='imagenet', 
                            input_shape=img[0].shape) 
        
        layer_outputs = [layer.output for layer in deepNN['model'][name].layers]
        
        deepNN['activations'][name] = models.Model(input=deepNN['model'][name].input, 
                                                  output=layer_outputs).predict(img)
        
        # add subplots
        print('adding subplot ',i)
        img_ax += [fig.add_subplot(gs1[0, i])]
        act_ax += [fig.add_subplot(gs2[0, i])]
        
        # remove tick marks
        img_ax[i].set_xticks([])
        img_ax[i].set_yticks([])
        act_ax[i].set_xticks([])
        act_ax[i].set_yticks([])  
        img_ax[i].set_title(name)
        act_ax[i].set_title('Filter Activation')
        # put an image and label in each subplot
        images += [img_ax[i].imshow(deepNN['activations'][name][0][0], animated=True)]
        labels += [img_ax[i].text(.50, 0.90, '', fontsize=14,
                                  horizontalalignment='center',
                                  transform=img_ax[i].transAxes, 
                                  color='white')]
  
        # add a filter activation images
        mask = np.zeros(shape=(1, deepNN['activations'][name][0].shape[3]))
        nodes += [act_ax[i].imshow(mask)]
        
        i += 1


    def layer_generator():
        
        # architectures
        architectures = deepNN['model'].keys()
        # Initialize dictionary of frame lists
        frame_data = {model_name: [] for model_name in architectures}
        model_count = len(deepNN.keys())
        
        # keep track of how many frames there are
        frame_data['count'] = {}
        frame_data['max'] = 0
        
        for name in architectures:
            layer_names = [l.name for l in deepNN['model'][name].layers]
            
            for ID, output in zip(layer_names, deepNN['activations'][name]):
                filter_count = output.shape[3]
        
                for x in range(filter_count):
                    
                    # image visualization
                    filter_image = output[0, :, : , x]
                    
                    # filter visualization
                    filter_mask = np.zeros(shape=(1, filter_count))
                     
                    if filter_count % 64 == 0:
                        filter_mask = filter_mask.reshape((-1, 64))
                    
                    # set index of visualized filter
                    filter_mask.itemset(x, 1)
                    
                    data = {'name':ID, 
                            'image': filter_image,
                            'filter': filter_mask, 
                            'model_count': model_count}
                    
                    frame_data[name].append(data)

            
            # Figure out what frames to incrememt by
            frame_data['count'][name] = len(frame_data[name])
            
            if frame_data['max'] < frame_data['count'][name]:
                frame_data['max'] = frame_data['count'][name]
                
        while True:
            data_packet = {}
            
            for f in range(frame_data['max']):
                for name in architectures:
                    idx =  f % frame_data['count'][name]
                    data_packet[name] = frame_data[name][idx]
                
                yield data_packet


    def plotter(args):
        k = 0
        for name, data_packet in args.items():
            images[k].set_data(data_packet['image'])
            labels[k].set_text('Layer ' + data_packet['name'])
            nodes[k] = act_ax[k].imshow(data_packet['filter'])
            k += 1
        res = images + labels + nodes

        return res
        
    anim = animation.FuncAnimation(fig, plotter,
                                   frames=layer_generator,
                                   interval=1,
                                   blit=True,
                                   repeat=False)
    
    # anim.save
    plt.show()
    
    