# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:22:49 2020

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
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

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


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 20)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


"""Saving the model"""

from sklearn.externals import joblib

joblib.dump(scaler, 'scalerDecisionTree.pkl')
joblib.dump(encoder, 'encoderDecisionTree.pkl')

joblib.dump(clf, 'classifierDecisionTree')



"""Plot the confusion matrix"""
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, normalize = None)

ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)

ax.set_ylabel('Animal classes')
ax.set_xlabel('')
ax.set_title('Confusion Matrix')
ax.yaxis.set_ticklabels(['bear', 'bird', 'cat'])
ax.xaxis.set_ticklabels(['', '', ''])

"""Creating a image of the Decision Tree"""
from sklearn import tree
fig, ax = plt.subplots(figsize=(20,20))
tree.plot_tree(clf, fontsize = 10)
plt.savefig('DecisionTree.png', dpi = 100)
