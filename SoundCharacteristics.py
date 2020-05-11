# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:59:52 2020

@author: kissf
"""

#import libraries
import librosa

import matplotlib.pyplot as plt
import librosa.display
import numpy as np

#Extracting spectral characteristics 

# 1.Zero Crossing Rate : The zero crossing rate is the rate of sign-changes along a signal, i.e., the rate at which the signal changes from positive to negative or back.
# Load the signal
x, sr = librosa.load('D:/Licenta/talkingtothemoon.wav')
#Plot the signal:
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
# Zoom on a specified zone
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()

#2. Spectral Centroid: The spectral centroid is a measure used in digital signal processing to characterise a spectrum. 
#It indicates where the center of mass of the spectrum is located. Perceptually, it has a robust connection with the impression of brightness of a sound.
import sklearn
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape
(775,)

# Creating a time variabile to view the audio signal
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

# Normalising the spectral centroid for visualisation
plt.figure(figsize=(14, 5))
def normalize(x, axis=0):return sklearn.preprocessing.minmax_scale(x, axis=axis)

#Realizing the plot of the Spectral Centroid
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')

#3. Spectral Rolloff
#Measure the shape of the signal. Reprezents the frequency below which a specified percentage of the total spectral energy lies
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')

#4. Spectral Bandwidth
#Wavelength interval in which a radiated spectral quantity is not less than half its maximum value.
spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]
plt.figure(figsize=(15, 9))
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
plt.plot(t, normalize(spectral_bandwidth_3), color='g')
plt.plot(t, normalize(spectral_bandwidth_4), color='y')
plt.legend(('p = 2', 'p = 3', 'p = 4'))


#4. Mel-Frequency Cepstral Coefficients : The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope.
x, fs = librosa.load('D:/Licenta/talkingtothemoon.wav')
librosa.display.waveplot(x, sr=sr)
mfccs = librosa.feature.mfcc(x, sr=fs)
print(mfccs.shape)
(20, 97)

#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')

#5. Chroma Frequencies : Chroma features are an interesting and powerful representation for music audio in which the entire spectrum is projected onto 12 
#bins representing the 12 distinct semitones (or chroma) of the musical octave.
# Loadign the file
x, sr = librosa.load('D:/Licenta/talkingtothemoon.wav')
hop_length = 512
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')