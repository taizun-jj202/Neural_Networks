import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap([ '#FF0000','00FF00', '#0000FF'])  #red, green , blue colours

iris = datasets.load_iris()
x, y =  iris.data, iris.target

xt,xtst, yt, ytst = train_test_split(x, y, test_size=0.2, random_state = 1234)


from knn import KNN
clf = KNN(5)
clf.fit(xt, yt)
predictions = clf.predict(xtst)

# Printing the accuracy
acc = np.sum(predictions == ytst) / len(ytst)
print(f"Accuracy : {acc}")