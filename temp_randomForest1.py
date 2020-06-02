#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 14:35:53 2019

@author: akshay.nadgire
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
X_train, X_test, y_train, y_test=train_test_split(load_breast_cancer().data, load_breast_cancer().target, test_size=0.2, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)
y_pred=model.predict(X_test)
conf_mat=confusion_matrix(y_pred=y_pred, y_true=y_test)
accuracy=np.sum(np.diag(conf_mat))/len(y_pred)

splits = KFold(n_splits=5, random_state=42)
acc = []
for train_index, test_index in splits.split(X=X_train, y=y_train):
    model = RandomForestClassifier()
    model.fit(X=X_train[train_index], y = y_train[train_index])
    y_pred=model.predict(X=X_train[test_index])
    acc.append(sum(y_train[test_index] == y_pred)/ len(y_pred))
