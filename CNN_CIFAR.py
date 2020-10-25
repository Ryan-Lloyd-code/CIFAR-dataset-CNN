# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 22:55:01 2020

@author: gilli
"""

import pandas as pd
import matplotlib.pyplot as plt

'Import CIFAR aniumal images dataset'
from tensorflow.keras.datasets import cifar10

'seperate data into training and test datasets'
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

'Find out the shape of the training data'

'50 000 images'
print(x_train.shape) 

'10 000 images'
print(x_test.shape)  

'display one of the images'
plt.imshow(x_train[0])

'''Try to normalize the image values by finding the max pixel value 
and divide the rest of the values by this max value'''

print(x_train.max())

x_train = x_train/225
x_test = x_test/255



'Importing the to_categorical utility in order to prevent the model from seeing this as a regression problem'
from tensorflow.keras.utils import to_categorical

'Finding out how many categories we are dealing with'
y_example = to_categorical(y_train)

'10 categories'
print(y_example.shape)

y_cat_train = to_categorical(y_train,10)
y_cat_test = to_categorical(y_test,10)


'MODEL CREATION'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

## FIRST SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

## SECOND SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 256 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(256, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

'Create early stopping function to optimize efficiency'
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',patience=3)
model.fit(x_train,y_cat_train,epochs=15,validation_data=(x_test,y_cat_test),callbacks=[early_stop])

'Plot the losses per iteration to track performance improvement'

losses = pd.DataFrame(model.history.history)

losses[['accuracy','val_accuracy']].plot()

losses[['loss','val_loss']].plot()

'Model final metrics'
print(model.metrics_names)
print(model.evaluate(x_test,y_cat_test,verbose=0))


'Display confusion matrix'
from sklearn.metrics import classification_report,confusion_matrix

predictions = model.predict_classes(x_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))







