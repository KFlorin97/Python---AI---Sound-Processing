# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:13:01 2020

@author: kissf
"""

#Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import librosa.display
import IPython.display
import pandas as pd
import random, warnings, os, pathlib, csv
#Keras
import keras
warnings.filterwarnings('ignore')
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv1D, Conv2D, Flatten, BatchNormalization, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import SGD
from keras.models import load_model
from keras.callbacks import EarlyStopping
from numpy import argmax
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Importing the dataset

#Analysing the Data in Pandas¶
data = pd.read_csv('dataset.csv')
data.head()

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)
#Encoding the Labels¶
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

#Scaling the Feature columns¶
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
#Dividing data into training and Testing set¶
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 29)

#ANN implementation
from keras import layers
from keras import layers
import keras
from keras.models import Sequential
model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

classifier = model.fit(X_train,
                    y_train,
                    epochs=500,
                    batch_size=128)

# save both the model and the scaler
keras.models.save_model(model,'classifierANN')
from sklearn.externals import joblib 

joblib.dump(scaler, 'scalerANN.pkl') 
joblib.dump(encoder, 'encoderANN.pkl') 

#Validating our approach¶
x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]



#Predictions on Test Data¶
predictions = model.predict_classes(X_test).flatten()

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions, normalize = None)

print(cm)

# classification report
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

"""Plot the confusion matrix"""


import seaborn as sns
import matplotlib.pyplot as plt

ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)

ax.set_ylabel('Animal classes')
ax.set_xlabel('')
ax.set_title('Confusion Matrix')
ax.yaxis.set_ticklabels(['bear', 'bird', 'cat'])
ax.xaxis.set_ticklabels(['', '', ''])

#Creating a image of the ANN
from ann_visualizer.visualize import ann_viz

ann_viz(model, title = "ANN", filename = "ANN")