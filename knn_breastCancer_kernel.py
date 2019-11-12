## Kernel KNN scratch code on Breast Cancer Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statistics

# Read the data
data = pd.read_csv("file:///E:/py_practice/bc_data.csv")
data = data.iloc[0:100,:]
x_data = data.drop(['id','diagnosis','Unnamed: 32'],axis=1)
y_data = data.diagnosis.map({'B':0,'M':1})

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.2)

## polynomial kernal ##
def poly_kernal(x,xt):
    return (1 + np.dot(x,xt))**2

## Gaussian kernal ##
def Gauss_kernal(x,xt):
    return (np.exp(-(np.sum(np.abs(x-xt)**2)/2)))

## sigmoid kernal ##
def sigmoid_kernal(x,xt):
    return (np.tanh(np.dot(x.T,xt) + 1))

# Applying all three functions
total_acc = []
for l in [poly_kernal, Gauss_kernal, sigmoid_kernal]:
    clas = []
    for i in range(len(x_test)):
        kernel_dist = []
        for j in range(len(x_train)):
            x = l(x_test.values[i,:],x_test.values[i,:].T)
            y = l(x_train.values[j,:],x_train.values[j,:].T)
            z = l(x_test.values[i,:],x_train.values[j,:].T)
            kernel_dist.append(np.sqrt(x**2 + y**2 - 2*z**2))
        clas.append(statistics.mode(y_train.values[np.argsort(kernel_dist)[range(3)]]))
    result = accuracy_score(clas,y_test)
    total_acc.append(result)

# print(total_acc)
print("Accuracy by using Polynomial Kernal is: ", total_acc[0]*100)
print("Accuracy by using Gaussian Kernal is: ", total_acc[1]*100)
print("Accuracy by using Sigmoid Kernal is: ", total_acc[2]*100)
