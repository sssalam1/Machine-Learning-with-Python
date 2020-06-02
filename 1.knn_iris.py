# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 19:48:47 2019

@author: Salam Saudagar
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_data = pd.read_csv("file:///E:/py_practice/Iris.csv")

sns.pairplot(iris_data.drop(labels=['Id'], axis = 1), hue='Species')

x_train, x_test, y_train, y_test = train_test_split(iris_data[['SepalLengthCm', 'SepalWidthCm',
                                                               'PetalLengthCm', 'PetalWidthCm']],
                                                    iris_data['Species'], random_state= 0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

pd.concat([x_test, y_test, pd.Series(y_pred, name='Predicted', index=x_test.index)], 
          ignore_index=False, axis=1)
print("Test set score: {:.2f}".format(knn.score(x_test, y_test)))
