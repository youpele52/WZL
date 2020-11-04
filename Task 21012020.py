#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:14:42 2020

@author: youpele
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# importing the dataset
dico = pd.read_json("dico_features.json")


#shuffling the rows 
#dico_ = dico.sample(frac=1)

#X2 = dico_.iloc[:,0:200]

X = dico.drop(['segment'], axis=1).values

# this  replaces the NaN data point with 0
X = pd.DataFrame(X).fillna(value = 0, ).values


y = dico.loc[:,'segment'].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier (n_neighbors=5, metric= 'minkowski', p=2 )
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# multilabel_confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix as mcm
multicm = mcm(y_test, y_pred)


#Accuracy Score
from sklearn.metrics import accuracy_score
accu_score = accuracy_score(y_test, y_pred)

# explained_variance_score
from sklearn.metrics import explained_variance_score
evs = explained_variance_score(y_test, y_pred)

# Classification report
from sklearn.metrics import classification_report
c_report = classification_report(y_test, y_pred)
