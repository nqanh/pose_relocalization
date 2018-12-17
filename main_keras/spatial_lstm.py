from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras.models import Model
from keras.layers import Flatten, Dense, Concatenate
import matplotlib.pyplot as plt

import sys
root_path = '/home/anguyen/workspace/paper_src/2018.icra.event.source'  # not .source/dataset --> wrong folder
sys.path.insert(0, root_path)

from dataset.data_io_cnn import load_data
from keras.callbacks import ModelCheckpoint, History 


import keras

import os

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed
from keras.models import Model, Sequential

from keras.utils import plot_model

from dataset.data_io_cnn_imu import load_data
from main_keras.vgg16 import VGG16


if __name__ == '__main__':
    img_model = VGG16(None)
    #print 'image model shape: ', img_model.shape
    #print 'image model shape: ', img_model.output_shape   # img_model is a MODEL not TENSOR
    #print 'model summary: ', img_model.summary()
    
    # Now let's get a tensor with the output of our vision model:
    img_input = Input(shape=(224, 224, 3))
    encoded_img = img_model(img_input)  # encoded_img is a TENSOR
    
    print 'encoded img shape: ', encoded_img.shape
    
    # try to add lstm
    for i in range(64):
        input = Input(shape=(64,1))
        lstm = LSTM(256, input_shape=(64,1), name='row_' + str(i)) (input)
        
    
    print 'ALL DONE!'
