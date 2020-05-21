# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:30:45 2020

@author: kissf
"""

import keras
import numpy as np
from sklearn.externals import joblib
import librosa
from sklearn.ensemble import RandomForestClassifier


#Importing the model
scaler = joblib.load('scalerDecisionTree.pkl') 
encoder= joblib.load('encoderDecisionTree.pkl') 

clf = joblib.load('classifierDecisionTree')

"""Predicting a sound"""

from keras.models import Sequential
model = Sequential()

songname= f'D:/Licenta/UnusedSounds/bird/bird-24.wav'

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
    
clf.predict([feature_list])
    
result = encoder.inverse_transform(clf.predict([feature_list]))
print("The selected sound is a " + result)