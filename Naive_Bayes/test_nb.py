import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib as plt

from nb import naive_bayes

def accuracy(y_true, y_pred):
    accuracy = np.sum((y_true == y_pred) / len(y_true))
    return accuracy

X, y = datasets.make_classification(n_samples= 1000, shuffle=True, n_features=10, random_state=123, n_classes=2)
xtr, xt, ytr, yt = train_test_split(X,y, test_size=0.2, random_state=123)

nb = naive_bayes()
nb.fit(xtr, ytr)
predictions = nb.predict(xt)

print(f"Naive bayes classification Accuracy : {accuracy(yt, predictions)}")