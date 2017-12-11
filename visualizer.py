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
    pass

def get_image(tensor=True):
    
    # Load in image, convert to tensor shape
    img = io.imread('britishBlue.jpg')
    img = img / 255
    if tensor:
        img = np.expand_dims(img, axis=0)
    return img


def create_plot_handles(model_list):
    
    plotter = {}
    image = get_image(tensor=False)
    
    # Animate the transformations
    plotter['fig'] = plt.figure(facecolor='grey', figsize=(12,6))
    plotter['gs1'] = gridspec.GridSpec(1, len(model_list))
    plotter['gs2'] = gridspec.GridSpec(1, len(model_list))
    
    plotter['gs1'].update(left=0.05, 
                          right=0.95, 
                          wspace=0.05, 
                          top=0.8, 
                          bottom=0.4)
    plotter['gs2'].update(left=0.05, 
                          right=0.95, 
                          wspace=0.05, 
                          top=0.35, 
                          bottom=0.25)
    
    plotter['ax1'] = {}
    plotter['ax2'] = {}
    
    for i, name in enumerate(model_list):
        
        # add subplots
        plotter['ax1'][name] = plotter['fig'].add_subplot(plotter['gs1'][0, i])
        plotter['ax2'][name] = plotter['fig'].add_subplot(plotter['gs2'][0, i])
        
        # remove tick marks
        plotter['ax1'][name].set_xticks([])
        plotter['ax1'][name].set_yticks([])
        plotter['ax1'][name].set_title(name)
        
        plotter['ax2'][name].set_xticks([])
        plotter['ax2'][name].set_yticks([])
        plotter['ax2'][name].set_title('Filter Activation')
        
        
        plotter['ax1'][name] = plotter['ax1'][name].imshow(image, animated=True)
        plotter['ax2'][name] = plotter['ax2'][name].imshow(np.ones((1,3)), animated=True)
        
    return plotter


def create_model_activations(model_list):
    
    deepNN = {'models'      : model_list,
              'activations': {},
              'layer_names': {},
              'count'      : len(model_list)}
    
    img = get_image(tensor=True)
    
    model_args = {'include_top' : False, 
                  'weights'     : 'imagenet', 
                  'input_shape' : img[0].shape}
    
    for name in model_list:
        # Create Model and activations
        model_network = getattr(applications, name)(**model_args)
        model_outputs = [layer.output for layer in model_network.layers]
        model_l_names = [layer.name for layer in model_network.layers]
        model_inshape = model_network.input
        
        deepNN['layer_names'][name] = model_l_names
        deepNN['activations'][name] = models.Model(input=model_inshape, 
                                                   output=model_outputs).predict(img)

    return deepNN


def create_filter_outputs(deepNN):
        
    # architectures
    model_list = deepNN['models']
    
    # Initialize dictionary of frame lists
    frame_data = {model_name: [] for model_name in model_list}
    
    # keep track of how many frames there are
    frame_data['count'] = {name : 0 for name in model_list}
    
    # Store name of models
    frame_data['models'] = model_list
    
    for name in model_list:
        for ID, output in zip(deepNN['layer_names'][name], 
                              deepNN['activations'][name]):
            
            filter_count = output.shape[-1]
    
            for x in range(filter_count):
                frame_data['count'][name] += 1
                
                # image visualization
                filter_image = output[0, :, : , x]
                
                # filter visualization
                filter_mask = np.zeros(shape=(1, filter_count))
                 
                if filter_count % 64 == 0:
                    filter_mask = filter_mask.reshape((-1, 64))
                
                # set index of visualized filter
                filter_mask.itemset(x, 1)
                
                data = {'layer': 'Layer' + ID + '\nfilter ' + str(x), 
                        'image': filter_image,
                        'filter': filter_mask}
                
                frame_data[name].append(data)
                
    frame_data['max'] = max(frame_data['count'].values())
    
    return frame_data


def layer_generator(frame_data):

        data_packet = {}
        model_list = frame_data['models']
        
        for f in range(frame_data['max']):
            for name in model_list:
                idx =  f % frame_data['count'][name]
                data_packet[name] = frame_data[name][idx]
            
            yield data_packet

def init(plot_handle):
    data_packet = {}
    
    for name in plot_handle['ax1'].keys():
        data_packet[name] = {'image' : get_image(tensor=False),
                             'layer' : 'input',
                             'filter': np.ones(shape=(1, 3))}
        
        ax_transform = plot_handle['ax1'][name].transAxes
        
        plot_handle['ax1'][name].imshow(data_packet[name]['image'], animated=True)
        plot_handle['ax1'][name].text(
                                    x=0.50, 
                                    y=0.860, 
                                    s=data_packet[name]['layer'], 
                                    fontsize=14, 
                                    horizontalalignment='center', 
                                    transform=ax_transform, 
                                    color='white')
        plot_handle['ax2'][name].imshow(data_packet[name]['filter'])
      
def animate(data_packet, plot_handle):
    frames = []
    
    for name in plot_handle['ax1'].keys():
        frames += [plot_handle['ax1'][name].set_data(data_packet[name]['image'])]
        frames += [plot_handle['ax1'][name].set_text(data_packet[name]['layer'])]
        frames += [plot_handle['ax2'][name].set_data(data_packet[name]['filter'])]
      
    return frames
    
    
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
    
    
    architectures = ['VGG16'] #, 'InceptionV3', 'Xception', 'ResNet50']

    plot = create_plot_handles(architectures)
    DNNs = create_model_activations(architectures)
    data = create_filter_outputs(DNNs)
    
    anim = animation.FuncAnimation(plot['fig'], animate,
                                   frames=layer_generator(data),
                                   init_func=init(plot),
                                   fargs=(plot,),
                                   interval=10,
                                   blit=True,
                                   repeat=False)
    
    # anim.save
    plt.show()
    
    