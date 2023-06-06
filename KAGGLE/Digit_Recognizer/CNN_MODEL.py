import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

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

class custom_call(Dataset):
    def __init__(self,img_df, label_df,dataset_csv):
        super().__init__() # Not necessary, adding this for base class initialization -> torch.utils.data.Dataset
        self.dataset  = dataset_csv
        self.img_df   = img_df
        self.label_df = label_df
        # Coverting them to tensors
        self.img_tnsr   = torch.tensor(self.img_df.values)
        self.label_tnsr = torch.tensor(self.label_df.values)

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        label = self.label_tnsr[idx]
        img   = self.img_tnsr[idx,:]
        return label,img


