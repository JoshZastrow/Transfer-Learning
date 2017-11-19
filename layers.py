# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 11:26:16 2017

@author: joshua zastrow

Note: Shape of Images are (num_train, height, width, channels)
"""

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import theano.tensor as T

class LRN(Layer):
    
    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)
        
    def call(self, x, mask=None):
        N, H, W, C = x.shape
        half_n = self.n // 2  # half the local region
        input_sqr = x ** 2
        extra_channels = np.zeros(shape=(N, H, W, int(C + 2*half_n)))
        # Set the center of the extra channels padded imput to be the squared
        # input
        input_sqr = T.set_subtensor(extra_channels[:, :, :, half_n:half_n + C],
                                    input_sqr)
        
        scale = self.k  # offset for scale
        norm_alpha = self.alpha / self.n
        
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, :, :, i : i + C]
        
        scale = scale ** self.beta
        x = x / scale
        return x
    
    def get_config(self):

        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
    

class PoolHelper(Layer):


    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:, 1:, 1:, :]


    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
#        
#class Convolution_layer(Layer):
#
#    """
#    A keras implementation of the forward pass for a convolutional layer.
#
#    Input:
#        nb_filter: Number of convolution filters to use.
#        nb_row: Number of rows in the convolution kernel.
#        nb_col: Number of columns in the convolution kernel
#        stride: The number of pixels between adjacent receptive fields in the
#        horizontal and vertical directions.
#        pad: The number of pixels that will be used to zero-pad the input.
#
#
#    """
#    def __init__(self, kernel_height=3, kernel_width=3, 
#                 nb_filters=64, stride=None, pad=None,
#                 **kwargs):
#
#        self.kernel_height = kernel_height
#        self.kernel_width = kernel_width
#        self.nb_filters = nb_filters
#        self.stride = stride
#        self.pad = pad
#        super(Convolution_layer, self).__init__(**kwargs)
#    
#    def build(self, input_shape):
#        
#        # Create a trainable weight variable
#        height, width, channels = input_shape[1]
#        self.W_shape = (self.nb_filters, 
#                        channels, 
#                        self.kernel_height, 
#                        self.kernel_width)
#        
#        self.W = self.add_weight(shape=self.W_shape,
#                                      name='{}_W'.format(self.name),
#                                      initializer='uniform',
#                                      trainable=True)
#        
#        self.b = self.add_weight((self.nb_filters), 
#                                 initializer='zero',
#                                 name='{}_b'.format(self.name))
#                                 
#        super(Layer, self).build(input_shape)
#        
#        
#    def call(self, x):
#        
#        x = K.expand_dims(x, 2)  # adds dummy dimension
#        
#        conv = K.conv2d(x, self.W, 
#                          strides=self.stride, 
#                          border_mode='valid',
#                          dim_ordering='tf')
#        
#        conv = K.squeeze(conv, 2)  # removes dummy dimension
#        conv += K.reshape(self.b, (1, 1, self.nb_filters))
#        
#        conv_relu = K.relu(conv, alpha=0., max_value=None)
#        conv_relu_pool = K.pool2d(output)
#        
#        return output
#    
#    def compute_output_shape(self, input_shape):
#        length = conv_output_length(input_shape[1],
#                                    self.filter_length,
#                                    self.border_mode,
#                                    self.subsample[0])
#        return (input_shape[0], length, self.nb_filter)
#    
#class Fully_Connected_layer(Layer):
#    
#    def __init__(self, output_dim, **kwargs):
#        self.output_dim = output_dim
#        super().__init__(**kwargs)
#        
#    def build(self, input_shape):
#        self.W = self.add_weight(name='{}_w'.format(name),
#                                 shape=(input_shape[1], self.output_dim),
#                                 trainable=True)
#        
#        self.b = self.add_weight(name='{}_b'.format(name),
#                                 shape=(input_shape[1], 1))
#        
#        super().build(self, input_shape)
#        
#    def call(self, x):
#        
#        output = K.dot(x, self.W)
#        output += self.b
#        
#        return output
#    
#    def compute_output_shape(self, input_shape):
#        return (input_shape[0], self.output_dim)
#                                 
                                
                                
        
        
        