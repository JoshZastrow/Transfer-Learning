# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:54:35 2017

@author: joshua
"""

import configparser
import argparse


def get_user_settings():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config


def create_parser():
    parser = argparse.ArgumentParser(description='Executes a '
                                                 'state of the art ML model '
                                                 'for feature extraction')

    parser.add_argument('-model', type=str, nargs='?',
                        required=False,
                        help='the model you would like to use for feature'
                             ' extraction. Current options:'
                             '\t-InceptionV3\n'
                             '\t-VGG16\n'
                             '\t-ResNet50\n'
                             '\t-Xception'
                             '\n\nleaving this blank will default to InceptionV3')

    parser.add_argument('-output', type=str, nargs='?',
                        required=True,
                        help='Output folder. Model name will be added'
                             '\n\nExample:\n\t'
                             'output/extracted-features')

    return parser


if __name__ == '__main__':

    config = configparser.ConfigParser()

    inputs = {
        'image dir': input('<str> Enter base directory path to image folder --> '),
        'log file path': input('<str> Enter file path to log file -->'),
        'batch size': input('<int> Enter number of images to read per batch process --> '),
        'sample size': input('<int> Enter total number of images to process --> '),
        'frame height': input('<int> Enter frame height of image sample --> '),
        'frame width': input('<int> Enter frame width of image sample -->'),
        'starting row': input('<int> Enter a sample index to skip to (default is 0) -->'),
        'output type': input('<str> Enter either hdf or folder for storage option-->')
    }

    config['DEFAULT'] = {
        'image dir': 'data',
        'log file path': 'data/interpolated.csv',
        'batch size': 5,
        'sample size': 20,
        'frame height': 480,
        'frame width': 640,
        'starting row': 0,
        'output type': 'folder'
    }

    config['USER'] = {}

    for key, val in inputs.items():
        if val:
            config['USER'][key] = val
        else:
            config['USER'][key] = config['DEFAULT'][key]

    with open('config.ini', 'w') as configfile:
        config.write(configfile)
