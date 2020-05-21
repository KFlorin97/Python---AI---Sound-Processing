# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:11:24 2020

@author: kissf
"""

songname= f'D:/Licenta/UnusedSounds/bird/bird-21.wav'

import keras
import numpy as np
from sklearn.externals import joblib
import librosa

"""Load the scaler and the model"""

scaler = joblib.load('scalerANN.pkl') 
encoder= joblib.load('encoderANN.pkl') 

model = keras.models.load_model('classifierANN')

""" Extracting the Spectrogram """
       
y, sr = librosa.load(songname, mono=True, duration=5)

feature_list = []
feature_list.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
feature_list.append(np.mean(librosa.feature.rms(y=y)))
feature_list.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
feature_list.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
feature_list.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
feature_list.append(np.mean(librosa.feature.zero_crossing_rate(y)))
mfcc = librosa.feature.mfcc(y=y, sr=sr)
  
for e in mfcc:
    feature_list.append(np.mean(e))
            
"""Create the test input """
X = np.array([feature_list])
X = scaler.transform(X)

prediction = model.predict_classes(X).flatten()
animal = encoder.inverse_transform(prediction)
print("The selected sound is a " + animal)


