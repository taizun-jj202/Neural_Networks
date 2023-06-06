# Linear regression is used to plot trends
# Statistical model that tries to show the relationship between
#     multiple/single input variables to find the value of 1 output variable
import numpy as np
import tqdm as tqdm

class LinearRgr :
    def __init__(self, lr = 0.001, n_iters = 1000, ):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        # init parameters
        n_samples , n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        # Implementing gradient descent here

        for _ in range(self.n_iters):
            # deriving all weights and bias by ourselves bcz implementing the algo in raw python
            y_pred = np.dot(X, self.weights)+self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            # Updating weights and biases
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred



