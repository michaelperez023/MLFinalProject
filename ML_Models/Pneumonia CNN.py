#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:45:23 2019

@author: michaelperez
"""

from ML_DataPipeline import DataPipeline
import _pickle as pickle
import joblib
import os

from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D

from keras import models

# starting point 
my_model= models.Sequential()

# Add first convolutional block
my_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', 
                    input_shape=(178,218,3)))
my_model.add(MaxPooling2D((2, 2), padding='same'))

# second block
my_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))
# third block
my_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))
# fourth block
my_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))

# global average pooling
my_model.add(GlobalAveragePooling2D())
# fully connected layer
my_model.add(Dense(64, activation='relu'))
my_model.add(BatchNormalization())
# make predictions
my_model.add(Dense(2, activation='sigmoid'))


# Show a summary of the model. Check the number of trainable parameters
my_model.summary()

# use early stopping to optimally terminate training through callbacks
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# save best model automatically
mc= ModelCheckpoint('yourdirectory/your_model.h5', monitor='val_loss', 
                    mode='min', verbose=1, save_best_only=True)
cb_list=[es,mc]


# compile model 
my_model.compile(optimizer='adam', loss='binary_crossentropy', 
                 metrics=['accuracy'])

path = r"C:\Users\michaelperez\Desktop\Compilation"
train_path = os.path.join(path, "train_images.pickle")
test_path = os.path.join(path, "test_images.pickle")
epochs = 30 # varied from 10-30 by 10
class_flag = True
labels = [0, 1]
seed = 42
cpu = joblib.cpu_count()

pipeline = DataPipeline("CNN", batch_size=24, train_file_path=train_path, test_file_path=test_path)
for epoch in range(epochs):
    if (epoch % 2) == 0:
        print("Epoch %d" %(epoch+1))
    if epoch + 1 == epochs:
        print("Epoch %d" %(epochs))
    iterations = pipeline.get_iterations()
    for iters in range(iterations):
        x_train, y_train = pipeline.get_training_batch(iters)
        """
        if class_flag:
            svm.partial_fit(x_train, y_train, classes=labels)
            class_flag = False
            continue
        svm.partial_fit(x_train, y_train)
        """

with open("Pneumonia CNN" + str(epochs) + "_.pickle", "ab") as model_file:
    pickle.dump(my_model, model_file)
    


"""

history = my_model.fit_generator(
        train_generator,
        epochs=30,
        steps_per_epoch=2667,
        validation_data=validation_generator,
        validation_steps=667, callbacks=cb_list)


# plot training and validation accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylim([.5,1.1])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("Custom_Keras_ODSC.png", dpi=300)


####### Testing ################################

# load a saved model
from keras.models import load_model
import os
os.chdir('yourdirectory')
saved_model = load_model('Custom_Keras_CNN.h5')

# generate data for test set of images
test_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Pictures/Celeb_sets/test',
        target_size=(178, 218),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)

# obtain predicted activation values for the last dense layer
test_generator.reset()
pred=saved_model.predict_generator(test_generator, verbose=1, steps=1000)
# determine the maximum activation value for each sample
predicted_class_indices=np.argmax(pred,axis=1)

# label each predicted value to correct gender
labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# format file names to simply male or female
filenames=test_generator.filenames
filenz=[0]
for i in range(0,len(filenames)):
    filenz.append(filenames[i].split('\\')[0])
filenz=filenz[1:]

# determine the test set accuracy
match=[]
for i in range(0,len(filenames)):
    match.append(filenz[i]==predictions[i])
match.count(True)/1000
"""

