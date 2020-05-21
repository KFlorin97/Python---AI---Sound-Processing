# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:43:23 2020

@author: kissf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import librosa.display
import IPython.display
import pandas as pd
import random
import warnings
import os
import pathlib
import csv
from numpy import argmax

""" Extracting the Spectrogram """
        
cmap = plt.get_cmap('inferno')
plt.figure(figsize=(8,8))
animals = 'bear bird cat'.split()
for animal in animals:
    i=0
    pathlib.Path(f'D:/Licenta/Images/{animal}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'D:/Licenta/Sounds/{animal}'):
        i=i+1
        songname = f'C:/Users/kissf/Desktop/Licenta/Sounds/{animal}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=5)
        print(animal + '-' + str(i))
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