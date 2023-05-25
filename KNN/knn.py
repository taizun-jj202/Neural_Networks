import numpy as np
from collections import Counter

def euclid_dist(x1,x2):
    return np.sqrt(np.sum(x1 - x2)**2)



class KNN:

    def __init__(self, k):
       self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, X):   # Takes in multiples samples so for 1 sample, we usehelper method _predict()
         predicted_labels = [self._predict(x) for x in X]
         return np.array(predicted_labels)

    def _predict(self, x):
        # compute distances
        distances = [euclid_dist(x, xt) for xt in self.x_train]

        # get k nearest samples/neighbours
        k_indices = np.argsort(distances)[:self.k] # Sliced upto 0...k to get k nearest samples
        k_nearest_label = [self.y_train[i] for i in k_indices]

        # majority vote, (i.e get the most common class label)
        most_common = Counter(k_nearest_label).most_common(1)
        return most_common[0][0]



