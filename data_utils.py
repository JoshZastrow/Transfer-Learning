from __future__ import absolute_import
from __future__ import print_function

# from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import os, math, time
import csv

plt.rcParams['figure.figsize'] = (20.0, 16.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class DataGenerator():
    """
    A data generator object that flows data from selected source.
    """
    def __init__(self, log_file, img_dir, 
                 batch_size=2, sample_size=10,
                 file_type='csv', img_ext = '.png',
                 target_size=(480, 640), starting_row=0):
        """
        args
        ----
            log_file: <str> path to the log file
            img_dir: <str> path to the image directory
            batch size: <int> how many samples to generate each time
            sample_size: <int> how many total samples to generate
            file_type: ['csv', 'h5'] log file type (.h5 not implemented)
        """        
        self.target_size = target_size
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.img_dir = img_dir
        self.img_ext = img_ext
        
        # check sample size against log row count
        with open(log_file,"r") as f:
                log = [csv.reader(f,delimiter = ",")]
                row = len(log)
                
                if row > self.sample_size:
                    print('Sample size larger than available data. '
                          'Setting sample size to {}'.format(row))
                    self.sample_size = row - 1
                    
        if file_type == 'csv':
            self.reader = pd.read_csv(log_file, 
                                      chunksize=batch_size, 
                                      header=0,
                                      skiprows=range(1, starting_row))
        else:
            raise ValueError('file type not implemented')
            
    def __iter__(self):
        for _ in range(0, self.sample_size, self.batch_size):
            batch = self.reader.get_chunk()
            images = self._process_images(batch.filename, self.img_ext)

            yield images, batch
        
    def __next__(self):
        batch = self.reader.get_chunk()
        images = self._process_images(batch.filename, self.img_ext)
        return images, batch
    

    def _process_images(self, dir_list, ext):
        """
        Loads images from file, performs image processing (if any)
        
        inputs
        ------
            dir_list: list of image file paths
        
        returns
        -------
            images: np array of images
        """
           
        images = []  # np.zeros(shape=(self.batch_size, *self.target_size, 3))
        
        for i, line in enumerate(dir_list):
            full_path = '/'.join([self.img_dir, line])
            full_path = os.path.splitext(full_path)[0] + ext

            if ext == '.npy':
                images.append(np.load(full_path))
            else:
                images.append(misc.imread(full_path, mode='RGB'))

            # TODO: Resize image to target size
            # TODO: Figure out how to use the image processing features
            #       of the inherited DataGenerator on the loaded image

            
        result = np.array(images)
        return result
    

class DataWriter(object):
    """
    writes numpy array to .h5 file. Creates a dataset group within root, then 
    a feature and label subgroup. 
    args
    ----
    folder: <str> filepath of the .h5 file
    dataset: <str> dataset name (name of model used to process data)
    """
    def __init__(self, output_dir, name, output_type='folder'):        
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.type = output_type    
        self.path = '/'.join([output_dir, name])

        # Create subfolders
        if self.type == 'hdf':
            self.path = self.path + '.h5'
            
        elif self.type == 'folder':
            for camera_view in ['left', 'center', 'right']:
                sub_folder = '/'.join([self.path, camera_view])
                
                if not os.path.exists(sub_folder):
                    os.makedirs(sub_folder)       
              
        else:
            raise ValueError("output type must be either <hdf> or <folder>")
            
    def __call__(self, x, y):
        
        if self.type == 'hdf':
            with pd.HDFStore(self.path) as hdf:
                for i, file in enumerate(y.filename):
                    hdf.put(file, x[i])

        else:
            for i, file in enumerate(y.filename):
                path = '/'.join([self.path, file])[:-4]
                np.save(path, x[i], allow_pickle=False)
         
            
def load_udacity_data(file_path='', 
                      img_dir='', 
                      batch=100, val_percent=.2,
                      shuffle=False, rescale=True):
    """
    loads in images as features, steering angle as label

    Inputs
    ----
    datasets : subfolder refering to bag folder (i.e 'HMB_ 1')
    batch : total number of samples to be read in
    val_percent: percent of batch to be assigned to validation (i.e 0.2)
    shuffle : TO DO -> shuffle dataset before returning

    returns
    ------
    X_train : (num_train, height, width, channels) array
    Y_train : (num_train, labels) array
    X_valid : (num_valid, height, width, channels) array
    Y_valid : (num_valid, labels) array
    """

    # Dataset folder
    if not file_path:
        file = "../Car/datasets/HMB_1/output/interpolated.csv"
        assert os.path.isfile(file), 'interpolated dataset not found'
    
    if not img_dir:
        img_dir = '../Car/datasets/HMB_1/output/'
        assert os.path.isdir(img_dir)
        
    # Starting with just center camera
    dataset = pd.read_csv(file)
    dataset = dataset[dataset['frame_id'] == 'center_camera']

    # Add directory path to dataset
    dataset['filename'] = img_dir + dataset['filename']

    # Setup data placeholders
    assert max(dataset['width']) == min(dataset['width'])
    assert max(dataset['height']) == min(dataset['height'])
    
    width = max(dataset['width'])
    height = max(dataset['height'])
    channels = 3

    if batch > dataset.shape[0]:
        batch = dataset.shape[0]

    X = np.zeros((batch, height, width, channels))
    Y = np.zeros((batch, ))

    num_train = int(batch * (1 - val_percent))
    num_valid = int(batch * (val_percent))
    
    mask = range(num_train, num_train + num_valid)
    X_valid = X[mask]
    Y_valid = Y[mask]
    
    mask = range(num_train)
    X_train = X[mask]
    Y_train = Y[mask]

    del X
    del Y

    count = 0

    # read in file data
    for rw in range(0, batch):

        angle = dataset['angle'].iloc[rw]
        ipath = dataset['filename'].iloc[rw]
        image = misc.imread(ipath)

        if count < num_train:
            X_train[count] = image
            Y_train[count] = angle
        else:
            X_valid[count % num_train] = image
            Y_valid[count % num_train] = angle

        count += 1

    data = {'X_train': X_train,
            'Y_train': Y_train,
            'X_valid': X_valid,
            'Y_valid': Y_valid}
    
    if rescale:
        data['X_train'] /= 255
        data['X_valid'] /= 255
        
    return data


def load_commai_data(log_file, cam_file):
    """
    loads .h5 files from comma AI's car dataset.

    Inputs
    ----
        log_file: file path for sensor log .h5 file
        cam_file: camera path for camera frames from .h5 file

    Returns
    -------
        log: Pandas Dataframe of log file, indexed with cam1_ptr
        cam: PyTables CArray of shape (frame, height, width, channels))

    """
    log_store = pd.HDFStore(log_file)
    cam_store = pd.HDFStore(cam_file)

    samples = len(log_store.root.cam1_ptr1)
    data_dic = {}

    # Read datasets into dictionary
    for d in log_store.root:
        if d.shape[0] == samples:
            if d.ndim == 1:
                data_dic[d.name] = d[:]
            else:
                for dim in range(d.shape[1]):
                    data_dic['{}-{}'.format(d.name, dim + 1)] = d[:, dim]

    # Average the log sensors in a Dataframe, create cam 4D array
    log = pd.DataFrame(data_dic).groupby('cam1_ptr').mean()
    cam = cam_store.root.X[:]

    return log, cam
       
if __name__ == "__main__":
    from config import get_user_settings
    
    # Get model settings from config file
    config = get_user_settings()['USER']
    
    dfeed = DataGenerator(log_file=config['log file path'],
                          img_dir=config['image dir'],
                          batch_size=int(config['batch size']), 
                          sample_size=100,
                          starting_row=int(config['starting row']))
    
    store = DataWriter('data-utils-output', 'sample_data')
    
    for imgs, labels in dfeed:    
        
        for i in range(imgs.shape[0]):
            for c in range(imgs.shape[3]):
                imgs[i,:,:,c] = imgs[i,:,:,c] / np.max(imgs[i,:,:,c])
            
        store(imgs, labels)

def stopwatch(start, comment):
    lap = math.floor(time.time() - start)
    print('{}: {} min {} sec'.format(comment, lap // 60, lap % 60))
    