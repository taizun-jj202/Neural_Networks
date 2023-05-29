import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Hyper parameters
lr =  0.001
batch_size = 4
num_epochs = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load and split dataset
train_dataset = datasets.CIFAR10(root='./dataCIFAR10',
                                 train=True,
                                 download=True,
                                 transform = transforms.ToTensor() )
test_dataset = datasets.CIFAR10(root='./dataCIFAR10',
                                 train=False,
                                 download=True,
                                 transform = transforms.ToTensor() )

train_loader = DataLoader( train_dataset,
                           batch_size=batch_size,
                           shuffle=True)
test_loader = DataLoader( test_dataset,
                           batch_size=batch_size,
                           shuffle=True)

classes = ( 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck')
# Building the model now
class CNN_NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5,5))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=100 )
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self,x):
        # x = F.relu(x)      # Activation func
        # x = self.pool(x)   # Pool layer
        # x = self.conv2(x)  #2 conv. layer
        # x = F.relu(x)      #2 activation  layer
        # x = self.pool(x)   #2 pool layer
        x = self.pool(F.relu(self.conv1(x)))  # Convolutional layer
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16 * 5 * 5)  # flatten to required size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

# Creating a model instance
model = CNN_NET().to(device)

# Optimizer and loss function for forward pass
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( model.parameters(), lr = lr)



# training the data over multiple epocs
def train_model():
    for epoch in range(num_epochs):
        for i, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            # forward pass
            out = model(data)
            loss = criterion(out, label)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
print("---------------------------------")
print("...Finished training the model...")
print("---------------------------------")

train_model()
# Calculating accuracy of model totally and per class also
with torch.no_grad():
    tot_correct = 0
    tot_samples = 0
    n_class_corr = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for data,label in enumerate(loader):
        data = data.to(device)
        label = label.to(device)
        out = model(data)
        _ , prediction = out.max(1)
        tot_correct += (prediction == label).sum().item()
        tot_samples += label.size(0)
        model_accuracy = (tot_correct / tot_samples)*100

        # Calculating accuracy of individual labels

        for i in range(batch_size):
            label = label[i]
            preds = preds[i]
            if(label[i] == preds[i]):
                n_class_corr[i] += 1
            n_class_samples[i] +=1

    print("---------------------------------")
    print(f"Accuracy of the model : {model_accuracy:.3f} %")
    print("---------------------------------")
    for i in range(10):
        class_accuracy = 100 * (n_class_corr[i]/n_class_samples[i])
        print(f"Accuracy of {classes[i]} Class == {class_accuracy}%")
    print("---------------------------------")




