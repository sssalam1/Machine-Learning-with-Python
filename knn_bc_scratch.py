## KNN scratch code on Breast Cancer Data
# Importing the libraries
import pandas as pd
import numpy as np
import statistics
from sklearn import preprocessing
# from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import BC data
data = pd.read_csv("file:///E:/py_practice/bc_data.csv")
data = data.iloc[0:100,:]
x_data = data.drop(['id','diagnosis','Unnamed: 32'],axis=1)
y_data = data.diagnosis.map({'B':0,'M':1})

x = preprocessing.normalize(x_data)

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x,y_data,test_size = 0.2) # x_train.shape

# Calculating all the distances for traning example and accuracy
total_acc = []
k = (1,3,5,7)
for v in k:
    clas = []
    for i in range(len(x_test)):
        d = []
        for j in range(len(x_train)):
            d.append(np.sqrt(sum((x_train[j,:]-x_test[i,:])**2)))
        clas.append(statistics.mode(y_train.values[np.argsort(d)[range(v)]]))
    
    accuracy = accuracy_score(y_test,clas)
    total_acc.append(accuracy*100)
#print(total_acc)

best_k = k[np.argsort(total_acc)[len(total_acc)-1]]
print("Best k is ", best_k)
