# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# load datatset
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale

X = load_iris().data
y = load_iris().target
X = scale(X)
from sklearn.svm import SVC

model = SVC(C=10,probability=True,gamma=0.0001)

from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test,y_pred)

acc= sum(y_test==y_pred)/len(y_pred)

from sklearn.metrics import accuracy_score,precision_score,recall_score,matthews_corrcoef

acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred,average='macro')
recal = recall_score(y_test,y_pred,average='macro')
mcc = matthews_corrcoef(y_test,y_pred)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import  make_scorer

score = make_scorer(accuracy_score)

clf = cross_val_score(estimator=model,X=X,y=y,cv=5,scoring=score)
