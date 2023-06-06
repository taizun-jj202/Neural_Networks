import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader,Dataset,random_split
import torch.nn.functional as F
from torchvision import transforms
import os
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper-parameters
learning_rate = 0.001
num_epochs = 5
batch_size = 24

# Writing a custom dataset
class custom_MNIST(Dataset):
    def __init__(self, train_csv , transforms = None):
        self.train_set = pd.read_csv(train_csv)
        self.transforms = transforms

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, item):
        imgs = (self.train_set.iloc[:, 1:])
        labels = (self.train_set.iloc[:,0])
        # if self.transforms:
        #     imgs = self.transforms(imgs)
        #     labels = torch.tensor(labels)

        return tuple(imgs, labels)


# Loading and splitting data
train_set = custom_MNIST(train_csv='train.csv',
                         transforms=transforms.ToTensor())

train_split, test_split = random_split(train_set,[33600, 8400])

train_loader = DataLoader(dataset=train_split,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_split,
                         batch_size=batch_size,
                         shuffle=True)


# Neural Network model:
class dig_rec(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=16,kernel_size=3, padding=1, stride=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(16*7*7, 392)
        self.fc2   = nn.Linear(392, 98)
        self.fc3   = nn.Linear(98, 10)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.pool(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool(x))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Model, loss func, optimizer
model = dig_rec()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# train function:
def train():

    print("---------------------------------------")
    print("Initializing model training")
    print("---------------------------------------")

    for epoch in range(num_epochs):
    #     print(f"Epoch :{epoch}/{num_epochs}")
        for idx,(imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()


    print("---------------------------------------")
    print("Finished model training")
    print("---------------------------------------")


train()