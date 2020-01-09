##################################
###### Image to FC2 vectors ######
##################################
### By using the VGG-16 pre-trained model ###
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import cv2
from keras.layers import Flatten, Dense, Input,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.models import Sequential
import tensorflow as tf


# Image to FC2 vector
def get_feature_vector(img, model):
    img = cv2.resize(img, (224, 224))
    
    feature_vector = model.predict(img.reshape(1, 224, 224, 3))
    # Handle the shape into (4096,)
    feature_vector = np.squeeze(feature_vector.T)
    
    return feature_vector



# Load & handle images 
def img2vector(paths, nr_houses):
    # Load VGG-16 for similarity
    vgg16 = keras.applications.VGG16(weights='imagenet', include_top= True, \
                                     pooling= 'max', input_shape=(224, 224, 3))
    
    model = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)
    FC2_out_length = 4096        # Output dimension of this VGG-16 model. 
    img_Vectors = np.zeros([FC2_out_length, nr_houses])
    for j, path in enumerate(paths):
        img = cv2.imread(path)
        img = img[...,::-1]
        vector = get_feature_vector(img, model)
        # Vectors here is a matrix, who has each img's FC2 in each column, and 100
        # columns have been stacked together to assemble this [img_Vectors] matrix.
        img_Vectors[:,j] = vector
        
    return img_Vectors


