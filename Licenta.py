# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:13:01 2020

@author: kissf
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf

from numpy import argmax
import librosa
import librosa.display
import IPython.display
import pandas as pd
import random
import warnings
import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
import keras
import warnings
warnings.filterwarnings('ignore')
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv1D, Conv2D, Flatten, BatchNormalization, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import SGD
import keras.backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping


""" Extracting the Spectrogram """
        
cmap = plt.get_cmap('inferno')
plt.figure(figsize=(8,8))
animals = 'bear bird cat'.split()
for animal in animals:
    pathlib.Path(f'D:/Licenta/Images/{animal}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'D:/Licenta/Sounds/{animal}'):
        songname = f'C:/Users/kissf/Desktop/Licenta/Sounds/{animal}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=5)
        print(y.shape)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=1024, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'D:/Licenta/Images/{animal}/{filename[:-3].replace(".", "")}.png')
        plt.clf()

"""
Converting audio into spectograms
"""
"""
Extracting features from Spectrogram and they are:
Mel-frequency cepstral coefficients (MFCC)(20 in number)
Spectral Centroid,
Zero Crossing Rate
Chroma Frequencies
Spectral Roll-off.
"""

header = 'filename chroma_stft ee spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

#We write the data to a csv file

file = open('dataset.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
animals = 'bear bird cat'.split()
for animal in animals:
    for filename in os.listdir(f'D:/Licenta/Sounds/{animal}'):
        songname = f'D:/Licenta/Sounds/{animal}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        rms = librosa.feature.rms(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {animal}'
        file = open('dataset.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
                    epochs=100,
                    batch_size=128)



#Validating our approach¶
x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]



#Predictions on Test Data¶
predictions = model.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions, normalize = False)