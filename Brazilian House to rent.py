# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 19:19:15 2021

@author: emamsur
"""
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#Import the dataset
data = pd.read_csv('houses_to_rent.csv')
data.shape
data.head()
data.drop(['Unnamed: 0'], axis =1)
data.info()
data['furniture'].value_counts().unique()
data['floor'].replace(to_replace='-',value=0,inplace=True)
data.head()
data.info()

#to replace the objects of a feature to int
data['furniture'].replace(to_replace='furnished', value=1, inplace=True)
data['furniture'].replace(to_replace='not furnished', value=0, inplace=True)
data['animal'].replace(to_replace='not acept',value=0,inplace=True)
data['animal'].replace(to_replace='acept',value=1,inplace=True)

#Removing the 'R$.' from few columns

for col in ["hoa","rent amount","property tax","fire insurance","total"]:
    data[col].replace(to_replace='R\$', value='', regex=True, inplace = True)
    data[col].replace(to_replace=',', value='', regex=True, inplace = True)

#Checking if all data is of integers and removing object values
data = data.astype(dtype=np.int64)
data.isin(['Sem info']).any()
data['hoa'].replace(to_replace='Sem info', value=1, inplace=True)
data.isin(['Sem info']).any()
data = data.astype(dtype=np.int64)
data.isin(['Incluso']).any()

data['hoa'].replace(to_replace='Incluso', value= 1 , inplace=True)
data['property tax'].replace(to_replace='Incluso', value= 1 , inplace=True)
data.isin(['Incluso']).any()
data = data.astype(dtype=np.int64)
data = data.sample(frac=1).reset_index(drop=True)

#Processing the train and test sets

Y = data['city']
X = data.drop('city', axis= 1)

#Applying ML Algorithm to train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

log_model = LogisticRegression(penalty='11', verbose=1)
log_model = LogisticRegression(C=0.01, penalty='l1',solver='liblinear')

svm_model = SVC(kernel='rbf', verbose=1)
nn_model = MLPClassifier(hidden_layer_sizes=(16,16), activation = 'relu', solver='adam', verbose=1)

log_model.fit(X_train,Y_train)
svm_model.fit(X_train,Y_train)
nn_model.fit(X_train,Y_train)

#Accuracy of the model

print(log_model.score(X_test, Y_test))
print(svm_model.score(X_test, Y_test))
print(nn_model.score(X_test, Y_test))

from sklearn.metrics import f1_score

print(f1_score(log_model, Y_test))
print(f1_score(svm_model, Y_test))
print(f1_score(nn_model, Y_test))
