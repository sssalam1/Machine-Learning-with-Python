# Feature selction

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

x, y = load_iris().data, load_iris().target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

model = RandomForestClassifier(n_estimators=100)
feature_select = SelectFromModel(model)

feature_select.fit(x_train, y_train)

x_train_transformed = feature_select.transform(x_train)

x_test_transformed = feature_select.transform(x_test)






from sklearn.feature_selection import chi2, SelectKBest

x, y = load_breast_cancer().data, load_breast_cancer().target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

feature_select_chi2 =  SelectKBest(chi2, k =10)
feature_select_chi2.fit(x_train, y_train)

x_train_transformed = feature_select_chi2.transform(x_train)

x_test_transformed = feature_select_chi2.transform(x_test)

from sklearn.feature_extraction import *
model = RandomForestClassifier(n_estimators = 500)

import sklearn.feature_selection as fs
feature_select_rfe = fs.RFECV(model, cv=5, min_features_to_select = 3)
model.fit(x_train, y_train)
model.feature_importances_
np.sort(model.feature_importances_)
np.argsort(model.feature_importances_)
