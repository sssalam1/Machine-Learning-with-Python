import numpy as np
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer

data0 = load_iris()
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
data_T = data1.T

##=== Applying Kernel Function
kern = poly_kernal(data1, data_T)

I = (1.0/150) * np.ones([150,150])
k_hat = np.mat(kern) - np.mat(I)*np.mat(kern) -np.mat(kern)*np.mat(I) + np.mat(I)*np.mat(kern)*np.mat(I)

#KL = np.dot(kern,I)
#LK = np.dot(I,kern)
#LKL = np.dot(LK,I)
#k_hat = kern - KL - LK + LKL

#print(k_hat[:,1].mean())

eig_val, eig_vec = np.linalg.eig(k_hat)

eig_val = np.real(eig_val)
eig_vec = np.real(eig_vec)

eig_norm = np.cumsum(eig_val)/sum(eig_val)

New_eig_vec = eig_vec[eig_norm<=0.99, :]
New_eig_vec = New_eig_vec.T


new_ex = []
for i in range(150):
    nr1 = np.dot(data1[i], data1.T)
    nr2 = np.dot(nr1, New_eig_vec)
    new_ex.append(nr2)

new_data = np.array(new_ex).reshape(New_eig_vec.shape)

#x_train, x_test, y_train, y_test = train_test_split(new_data,clas,test_size=0.2, random_state=42)

model = RandomForestClassifier()
#model.fit(x_train, y_train)
#y_pred=model.predict(x_test)
#accuracy = accuracy_score(y_pred,y_test)
#accuracy
score = make_scorer(accuracy_score)
cvs = cross_val_score(estimator=model, X= new_data, y = clas, cv = 5, scoring=score )
#print("Accuracy is", accuracy*100)



#Mean=New_eig_vec.mean(axis=1)
#Sigma=New_eig_vec.std(axis=1)
#std_eigen_vec=(New_eig_vec-Mean)/Sigma
#std_eigen_vec=std_eigen_vec.T
#
#std_eigen_vec2 = scale.fit_transform(New_eig_vec)
#std_eigen_vec2=std_eigen_vec2.T
#
#new_data = np.transpose(np.dot(np.transpose(std_eigen_vec),kern))


