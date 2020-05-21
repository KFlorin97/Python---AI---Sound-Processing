# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:28:50 2020

@author: kissf
"""

from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
import librosa
import keras
import matplotlib.pyplot as plt

#Load dataset
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

#Spliting data into test and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 29)


#Create the Gaussian Classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 1000)

#Train the model using the training sets, y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#Testing the Accuracy
from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

#Saving models

from sklearn.externals import joblib
joblib.dump(scaler, 'scalerRandomForest.pkl')
joblib.dump(encoder, 'encoderRandomForest.pkl')

joblib.dump(clf, 'classifierRandomForest')



#Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, normalize = None)

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

