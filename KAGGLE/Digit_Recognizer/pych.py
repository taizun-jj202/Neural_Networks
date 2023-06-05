import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader


# Setting device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading train and test csv files
train_set = pd.read_csv('train.csv')
test_set  = pd.read_csv('test.csv')
print("train_set dimensions: {}".format(train_set.shape))
print("test_set dimensions : {}".format(test_set.shape))


# Converting variables to tensor
train_set = torch.tensor(train_set.values)
test_set = torch.tensor(test_set.values)

# Separating pixel data and labels from train_set
imgs = train_set[:,1:]
imgs.to(device)
labels = torch.unsqueeze(train_set[:,0],dim=1)
labels.to(device)
# print(labels.shape)

# CNN model for classification:
class  dig_reg(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))  # MaxPool to extract most prominent features of the image and also downsample image
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=36, kernel_size=(3, 3), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))  # MaxPool to remove translation errors(i.e even in features move, to this that is not a concern
        self.fc1 = nn.Linear(in_features=28 * 28 * 36, out_features=2000)
        self.fc2 = nn.Linear(in_features=2000, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.pool1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool2(x))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Hyper-parameters
learning_rate = 0.001
num_epochs = 5
batch_size = 64


# Creating instance of NN, loss func, optimizer
model = dig_reg()
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Dividing model into batches
train_batches = DataLoader(dataset=train_set,
                           batch_size=batch_size,
                           shuffle=True)


# Creating training function for the model:
def model_train():

    print("-----------------------------------------------")
    print("Initializing model training")
    print("-----------------------------------------------")
    for epoch in range(num_epochs):
        for i in train_batches:
            # imgs = batch[:, 1:].to(device)
            imgs = train_set[:, 1:].reshape(-1, 1, 28, 28)
            labels = torch.unsqueeze(train_set[:, 0], dim=1).to(device)
            print(labels)
            imgs, labels = imgs.to(device, dtype=torch.float32), labels.to(device, dtype = torch.long )

            loss = loss_fn(model(imgs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

model_train()


# Calculating model accuracy
def model_acc():

    tot_samples = 0
    tot_correct = 0

    for batch in train_batches:
        imgs = train_set[:,1:].reshape(-1,1,28,28) #Get batches of 28*28 pictures



# model_train()