#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:45:23 2019

@author: michaelperez
"""

from ML_DataPipeline import DataPipeline
import os
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import models, regularizers

# starting point 
my_model= models.Sequential()

# Add first convolutional block
my_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', 
                    input_shape=(750, 750, 1)))
my_model.add(MaxPooling2D((2, 2), padding='same'))

# second block
my_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))
# third block
#my_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#my_model.add(MaxPooling2D((2, 2), padding='same'))
# fourth block
#my_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#my_model.add(MaxPooling2D((2, 2), padding='same'))

# global average pooling
my_model.add(GlobalAveragePooling2D())
# fully connected layer
my_model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
my_model.add(BatchNormalization())
# make predictions
my_model.add(Dense(1, activation='sigmoid'))


# Show a summary of the model. Check the number of trainable parameters
my_model.summary()

# use early stopping to optimally terminate training through callbacks
#from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
#es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# save best model automatically
#mc= ModelCheckpoint('yourdirectory/your_model.h5', monitor='val_loss', 
#mode='min', verbose=1, save_best_only=True)
#cb_list=[es,mc]


# compile model 
my_model.compile(optimizer='sgd', loss='binary_crossentropy', 
                 metrics=['accuracy'])

path = r"/Users/michaelperez/Desktop/Compilation/"
train_path = os.path.join(path, "train_images.pickle")
test_path = os.path.join(path, "test_images.pickle")
epochs = 1 # varied from 10-30 by 10

pipeline = DataPipeline("CNN", batch_size=32, train_file_path=train_path, 
                        test_file_path=test_path)
for epoch in range(epochs):
    if (epoch % 2) == 0:
        print("Epoch %d" %(epoch+1))
    if epoch + 1 == epochs:
        print("Epoch %d" %(epochs))
    iterations = pipeline.get_iterations()
    for iters in range(iterations):
        x_train, y_train = pipeline.get_training_batch(iters)
        my_model.fit(x_train, y_train, epochs = 1, batch_size = 32)
    
my_model.save('CNNClassifier_1_.h5')  # creates a HDF5 file 'CNNClassifier_1_.h5'
    
