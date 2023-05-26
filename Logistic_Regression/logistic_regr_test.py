import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib as plt
from logistic_regression import log_regr

data = datasets.load_breast_cancer()
x,y = data.data, data.target

xtr, xt, ytr, yt = train_test_split(x,y ,test_size=0.2, shuffle=True)

def accuracy(y_true, y_predicted):
    acc = np.sum((y_true == y_predicted) / len(y_true))
    return acc

regr = log_regr(lr=0.001, n_iters=1000)
regr.fit(xtr, ytr)
predictions = regr.predict(xt)
print("Logistic Regression accuracy : ", accuracy(yt,predictions))