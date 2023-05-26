# Using KNN to classify the MNIST dataset
# Using the pytorch library for this just for experimentation

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from knn import KNN # Importing my KNN model

# Hyper-parameters for pytorch dataloader
batch_size = 64

# importing the training and testing datasets
train_data = datasets.MNIST(root='./datasetMNIST',
                            train=True,
                            download=True)
test_data = datasets.MNIST(root='./datasetMNIST',
                           download=True,
                           train=False)
# train, test_loader equivalent to x_train, x_test
train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          shuffle=True )

test_loader = DataLoader(dataset=test_data,
                         shuffle=True,
                         batch_size=batch_size)

# Extracting data, index from the train_data dataloader
for i in range(len(train_loader)):
    data_tr, index_tr = enumerate(train_loader)

# print(data_tr, index_tr)
# # passing the training data into the KNN model
# k=3
# model = KNN(k)
# model.fit(train_loader)
