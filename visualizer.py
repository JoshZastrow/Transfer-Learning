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
from model import FeatureGenerator

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
