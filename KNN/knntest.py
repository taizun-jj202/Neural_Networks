import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap([ '#FF0000','00FF00', '#0000FF'])  #red, green , blue colours

iris = datasets.load_iris()
x, y =  iris.data, iris.target

xt,xtst, yt, ytst = train_test_split(x, y, test_size=0.2, random_state = 1234)


# from knn import KNN
# clf = KNN(5)
# clf.fit(xt, yt)
# predictions = clf.predict(xtst)

# Printing the accuracy
# acc = np.sum(predictions == ytst) / len(ytst)
# print(f"Accuracy : {acc * 100} %")

# Plotting a graph for accuracy vs 'k'
from knn import  KNN

k_values = []
accuracy_values = []
for k in range(1, 21,1):

    model = KNN(k)
    model.fit(xt,yt)
    predictions = model.predict(xtst)
    accuracy = np.sum(predictions == ytst) / len(ytst) #Checking the predicted label and actual label (ytst is actual label)
    k_values.append(k)
    accuracy_values.append(accuracy)


# Plotting kvalues vs accuracy_values
plt.plot(k_values, accuracy_values)
plt.xlabel("Accuracy")
plt.ylabel("'k'")
plt.title("Optimal Values of 'k' to Accuracy")
plt.show()
