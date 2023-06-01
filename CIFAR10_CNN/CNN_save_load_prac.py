import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Setting model to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setting hyperparameters
learning_rate = 0.001
num_epochs = 5
batch_size = 128


# Importing and  splitting data
train_data = datasets.CIFAR10(root='./CIFAR10',
                               transform=transforms.ToTensor(),
                               download=True,
                               train=True)
test_data = datasets.CIFAR10(root='./CIFAR10',
                              transform=transforms.ToTensor(),
                              download=True,
                              train=True)
    # Splitting data into batches using dataloaders
train_loader = DataLoader( dataset=train_data,
                           batch_size=batch_size,
                           shuffle=True)
test_loader = DataLoader( dataset=test_data,
                          batch_size=batch_size,
                          shuffle=True)

classes = ('airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck')

# Creating CNN for classification
class CNN_CIFAR(nn.Module):

    '''Out of Convolutional layers:
            O = (i - k  + 2p)/s + 1

        Out of Maxpool layer:
            O = (i-k)/s + 1
    '''
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d( in_channels=3, out_channels=9, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.pool1 = nn.MaxPool2d( kernel_size=(1,1), stride=(1,1))  # MaxPool to extract most prominent features of the image and also downsample image
        self.conv2 = nn.Conv2d( in_channels=9, out_channels=36,kernel_size=(3,3), stride=(1,1))
        self.pool2 = nn.MaxPool2d( kernel_size=(1,1), stride=(1,1))  # MaxPool to remove translation errors(i.e even in features move, to this that is not a concern
        self.fc1 = nn.Linear(in_features=30*30*36,out_features= 972)
        self.fc2 = nn.Linear(in_features=972 , out_features=162)
        self.fc3 = nn.Linear(in_features=162 , out_features=10)

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

# Creating instance of model
model = CNN_CIFAR().to(device)

# Optimizer and loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# Saving and loading previous state/weights of model
checkpoint_params = {'state': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
def save_checkpoint(state, file='state.pth.tar'):
    print("--------- Saving checkpoint ---------")
    torch.save(state, file)

def load_checkpoint(checkpoint_params):
    print("--------- Loading saved parameters ---------")
    model.load_state_dict(checkpoint_params['state'])
    optimizer.load_state_dict(checkpoint_params['optimizer'])


# Training and Validation of model
def train():
    print("------------------------------------")
    print("...Executing model training...")
    print("------------------------------------")
    for epoch in range(num_epochs):
        for idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)

            # score = model(data)
            loss = loss_fn(model(data), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # save_checkpoint(checkpoint_params)
    print("------------------------------------")
    print("...Finished training the model...")
    print("------------------------------------")


def validate():
    n_samples = 0
    n_correst = 0
    n_samples_corr = [0 for _ in range(10)]
    n_correst_class= [0 for _ in range(10)]

    for (data, label) in test_loader:
        data, label = data.to(device), label.to(device)

        _ , preds = model(data).max(1) # Returns tuple (data, label), were interested only in the predicted label
        n_correst += (preds == label).sum().item()
        n_samples += label.size(0)

        acc = n_correst/n_samples
    print(f"Overall model Accuracy == {acc*100}%")


    for i in range(min(batch_size, len(label))):
        labels = label[i]
        pred = preds[i]
        if (labels == pred):
            n_correst_class[labels] +=1
        n_samples_corr[labels] +=1

    print("----------------------------------------")
    for i in range(10):
        class_acc = n_correst_class[i] / n_samples_corr[i]
        print(f"Accuracy of Class {classes[i]} == {class_acc*100}%")
    print("----------------------------------------")


train()
validate()







