# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 22:37:07 2019

@author: Salam Saudagar
"""
###================ Kernel PCA For Breast Cancer Data ===============###

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer

data0 = load_breast_cancer()
data = data0.data
feature = data0.feature_names
clas = data0.target

## polynomial kernal ##
def poly_kernal(x,xt):
    return (1 + np.dot(x,xt))**2

## Gaussian kernal ##
def Gauss_kernal(x,xt):
    return (np.exp(-(np.sum(np.abs(x-xt)**2)/2)))

##=== Standerdizing the data
scale = preprocessing.StandardScaler()
data1 = scale.fit_transform(data)

##=== Applying Kernel Function
kern = poly_kernal(data1, data1.T)

##=== Normalizing the Kernel Function
I = (1.0/data.shape[0]) * np.ones([data.shape[0],data.shape[0]])
k_hat = np.mat(kern) - np.mat(I)*np.mat(kern) -np.mat(kern)*np.mat(I) + np.mat(I)*np.mat(kern)*np.mat(I)

##=== Calculating Eigen Values and Eigen vectors 
eig_val, eig_vec = np.linalg.eig(k_hat)

eig_val = np.real(eig_val)
eig_vec = np.real(eig_vec)

##=== Selecting the Eigen-Vectors by using Eigen values
eig_norm = np.cumsum(eig_val)/sum(eig_val)

#New_eig_vec = eig_vec[eig_norm<=0.96, :]
#New_eig_vec = New_eig_vec.T
New_eig_vec = eig_vec[:, eig_norm<=0.96]

##=== Creating the New data by taking dot products of eigen vector and scaled data
new_ex = []
for i in range(data.shape[0]):
    nr1 = np.dot(data1[i], data1.T)
    nr2 = np.dot(nr1, New_eig_vec)
    new_ex.append(nr2)

new_data = np.array(new_ex).reshape(New_eig_vec.shape)

##=== Applying Random Forest Model
model = RandomForestClassifier(n_estimators = 100)
score = make_scorer(accuracy_score)
cvs = cross_val_score(estimator=model, X= new_data, y = clas, cv = 5, scoring=score )
print(cvs)