# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 21:23:37 2018

@author: ipist
"""
 
 
from keras.models import Sequential
from keras.layers import Convolution2D #1st step of making CNN, convolution step
from keras.layers import MaxPooling2D #pooling step
from keras.layers import Flatten #flattening step.  Convert pooling step
from keras.layers import Dense #fully connected layer
    
classifier = Sequential ()

train_data = 'images/train'

#cnn is still a sequence 
#each package= 1 step in construction

#*****************building the cnn******************************
#initializing a CNN
classifier = Sequential ()
#convolution - max pooling - flattening - full connection
#we create many feature maps to obtain our 1st convolution layer

#step 1 - convolution
#32 feature detectors 3x3 dimensions 
#same=default value
#Shape of input image - convert all images into one same single format.  3d array - 3= # of channels.  1=black and white.  3=colored.  
# colored images - 256x256
#(3, 64, 64))) used instead of (3,256,256))) need smaller size
#activation function, rectifier
#relu removes negative pixels.  
classifier.add(Convolution2D(32,3,3, border_mode='same', input_shape=(64, 64, 3), activation='relu'))
#max pooling - reduces # of nodes.  if we don't reduce, we get too large of a vector, and model will be too compute intensive.

#step 2 pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(32,3,3, border_mode='same', input_shape=(64, 64, 3), activation='relu'))
#max pooling - reduces # of nodes.  if we don't reduce, we get too large of a vector, and model will be too compute intensive.

#step 2 pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#size of feature maps now divided by 2

#step 3 flattening
classifier.add(Flatten())#keras understands #single vector is now created.  Now we create a classic ANN to classify the images

#step 4 full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
#output dim = # of nodes in hidden layer.  choose  between input nodes and output nodes.  no rule of thumb, ~100 is a good choice.
#best pick a power of 2
#output layer
#one class predicted
classifier.add(Dense(output_dim = 1   , activation = 'softmax'))#sigmoid function because binary.  SOftmax when multiple
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics=['accuracy'])  #logarithmic loss , binary outcome., more than 2 outcomes, 
#we could need categorical crossentropy


#********************fitting CNN to images************************************************
#image augmentation trick = enriches dataset without adding more images
#from https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        train_data,
        target_size=(64,64),#size of images expected in cnn model same as input shape
        batch_size=32, #size of batches
        class_mode='binary') #binary or more than 2 categories.  

#test_set = test_datagen.flow_from_directory(
#        'dataset/test_set',
#        target_size=(64, 64),#same as input shape
#        batch_size=32,
#        class_mode='binary')

classifier.fit_generator(
        training_set, 
        steps_per_epoch=10222,# num of images in training set
        epochs=25), #lower number = shorter waiting time
       # validation_data=test_set, 
       # validation_steps=2000)
 
#fit cnn to dataset and train performances.  