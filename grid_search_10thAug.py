# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 14:24:34 2019
@author: Salam Saudagar
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RandomizedSearchCV, train_test_split , GridSearchCV
from sklearn.ensemble import RandomForestClassifier

data0 = load_breast_cancer()
x = data0.data
feature = data0.feature_names
y = data0.target

x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.2, random_state = 42)

random_grid = { 'max_depth' : [10, 20],
               'n_estimators' : [50, 100] }

###===== Using RandomizedSearchCV
rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator=rf , param_distributions= random_grid, n_iter=4, cv = 5, verbose=2,
                               random_state=42, n_jobs = -1)
rf_random.fit(x_train , y_train)
print(rf_random.best_params_)

###===== Using GridSearchCV
rf = RandomForestClassifier()

rf_random = GridSearchCV(estimator=rf , param_grid= random_grid, cv = 5, n_jobs = -1)
rf_random.fit(x_train , y_train)
print(rf_random.best_params_)