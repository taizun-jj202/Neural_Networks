import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd

class dig_rec(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,kernel_size=3, padding=1, stride=1)
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

class CNN(nn.Module):

    def __init__(self,input_size = 1, num_classes = 10):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes) #Fully conncected layer

    def forward(self,x) :
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

class lineNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=392)
        self.fc2 = nn.Linear(in_features=392, out_features=196)
        self.fc3 = nn.Linear(in_features=196, out_features=49)
        self.fc4 = nn.Linear(in_features=49, out_features=10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

class custom_call1(Dataset):
    def __init__(self,img_df, label_df,dataset_csv):
        super().__init__() # Not necessary, adding this for base class initialization -> torch.utils.data.Dataset
        self.dataset = dataset_csv
        self.img_df = dataset_csv.iloc[:, 1:]
        self.label_df = dataset_csv.iloc[:, 0]
        # self.img_df = img_df
        # self.label_df = label_df
        # Coverting them to tensors
        self.img_tnsr = torch.tensor(self.img_df.values)
        self.label_tnsr = torch.tensor(self.label_df.values)

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        label = self.label_tnsr[idx-2]
        img = self.img_tnsr[idx-2]
        img = img.reshape(-1, 1, 28,28)
        return label,img
import torch
from torch.utils.data import Dataset
from PIL import Image

class custom_call(Dataset):
    def __init__(self, dataset_csv, transform=None):
        super().__init__()
        self.dataset = dataset_csv
        self.img_df = dataset_csv.iloc[:, 1:]
        self.label_df = dataset_csv.iloc[:, 0]
        self.img_tnsr = torch.tensor(self.img_df.values, dtype=torch.float32)
        self.label_tnsr = torch.tensor(self.label_df.values)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        label = self.label_tnsr[idx-2]
        img = self.img_tnsr[idx-2]
        img = img.reshape(28, 28)

        # Convert tensor to PIL Image
        img_pil = Image.fromarray(img.numpy(), mode='L')

        if self.transform:
            img_pil = self.transform(img_pil)

        img_tensor = torch.tensor(img_pil)

        img_tensor = img_tensor.unsqueeze(0)
        return label, img_tensor

