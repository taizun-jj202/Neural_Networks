import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import random_split,DataLoader,Dataset
from MODEL import *

train_csv = pd.read_csv('train.csv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device in Use: {device}")
print(f"Length of train.csv : {len(train_csv)}")

train_data = custom_call(train_csv)
train_loader = DataLoader(dataset=train_data,
                          batch_size=24,
                          shuffle=True)
test_loader = DataLoader(dataset=train_data,
                         batch_size=24,
                         shuffle=True)

model = nnmodel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# training loop
def train():
    print("... Training model ...")
    for i,l in train_loader:
        i, l = i.to(device), l.to(device)

        output = model(i)
        optimizer.zero_grad()
        loss = loss_fn(output, l)
        loss.backward()
        optimizer.step()
    print("... Finished training ...")

# Finding accuracy of model
def acc():
    tot_correct = 0
    tot_samples = 0
    print("Calculating accuracy of the model....")
    for i,la in test_loader:
        i, la = i.to(device), la.to(device)

        with torch.no_grad():
            _, predicted = model(i).max(1)
            tot_correct += (predicted == la).sum().item()
            tot_samples += la.size(0)

    acc = tot_correct/tot_samples
    print(f"Accuracy: {acc*100} %")

def save():
    torch.save(model.state_dict(), './v6_wb.pth')
    print('... Saving the model weights ...')

def load():
    model.load_state_dict(torch.load('./v6_wb.pth'))
    print("... Loading the model ...")

# train()
# save()
load()
acc()




