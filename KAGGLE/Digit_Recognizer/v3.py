import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_set = pd.read_csv('train.csv')
train_set =torch.tensor(train_set.values)
print("Size of train_set: {}".format(train_set.shape))
test_set = pd.read_csv('test.csv')
test_set = torch.tensor(test_set.values)
print("Size of train_set: {}".format(test_set.shape))

# load data in lenghts of 1750 samples
train_data = DataLoader(batch_size=24,
                        shuffle=True,
                        dataset=train_set)
print("Samples in each batch of data : {}".format(enumerate(train_data)))
# Create Neural NETWORK model
list  = [1,2,3,4]

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

# Creating model, loss fnc, optimizer
model = dig_rec()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train model
def train():

    print("---------------------------------------")
    print("Initializing model training")
    print("---------------------------------------")

    for sample in train_data:
        imgs = sample[:,1:].to(device,dtype=torch.long)
        labels = sample[:,0].to(device, dtype=torch.long)

        imgs = imgs.reshape(-1,1, 28,28)
        predicted_labels = model(imgs)
        loss = loss_fn(predicted_labels, labels)

        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

    print("---------------------------------------")
    print("Finished model training")
    print("---------------------------------------")
    torch.save(model.state_dict(), './model_weights.pth')
    print("Saving the model...")



# Finding accuracy of the model
def acc():
    tot_samples = 0
    tot_correct = 0

    print("---------------------------------------")
    print("Checking model accuracy...")
    with torch.no_grad():
        for sample in train_data:
            imgs = sample[:, 1:].to(device, dtype=torch.long)
            labels = sample[:, 0].to(device, dtype=torch.long)


            imgs = imgs.reshape(-1,1,28,28)
            predicted = model(imgs).max(1)
            tot_correct += (predicted == labels)
            tot_samples += len(labels)

    acc = tot_correct/tot_samples
    print("Accuracy : {}".format(acc*100))

# acc()

if __name__ == '__main__':
    # # train()
    # for params in model.state_dict():
    #     print(params , "\t", model.state_dict()[params].size())

    print("\nLOading saved model for accuracy test")
    model.load_state_dict(torch.load('./model_weights.pth'))
    acc()