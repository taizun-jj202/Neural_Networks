import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# Importing and splitting the dataset
X, y = datasets.make_regression(n_samples=100, n_features=1, noise = 20, random_state = 4)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
# plt.show()
# print(xt.shape)
# print(yt.shape)

from linear_regression import LinearRgr
regressor = LinearRgr(lr = 0.01)
regressor.fit(x_train, y_train)
predicted = regressor.predict(x_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

mse = mse(y_test, predicted)
print(mse)

y_pred_line = regressor.predict(X)
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(x_train, y_train)
m2 = plt.scatter(x_test, y_test)
plt.plot(X, y_pred_line,color = 'black', linewidth = 2, label="Prediction")
plt.show()