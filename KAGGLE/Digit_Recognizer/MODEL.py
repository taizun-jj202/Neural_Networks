import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms

class custom_call(Dataset):

    def __init__(self,train_csv):
        self.train_data = train_csv
        self.pixels = train_csv.drop('label',axis=1)
        self.labels = train_csv['label']


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        img = self.pixels.iloc[item-2].values
        labl = self.labels.iloc[item-2]
        img_tnsr = torch.tensor(img)
        labl_tnsr= torch.tensor(labl)
        return img_tnsr,labl_tnsr

class nnmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=392)
        self.fc2 = nn.Linear(in_features=392, out_features=196)
        self.fc3 = nn.Linear(in_features=196, out_features=49)
        self.fc4 = nn.Linear(in_features=49, out_features=10)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x
