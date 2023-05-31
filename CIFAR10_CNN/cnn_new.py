import torch
import torch.nn as nn
import torchvision.models
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# setting model to gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5


model = torchvision.models.vgg16(pretrained = True)
# print(model)
for param in model.parameters():
    param.requires_grad = False


model.classifier = nn.Sequential(nn.Linear(25088, 512),
                                 nn.ReLU(),
                                 nn.Linear(512,100),
                                 nn.ReLU(),
                                 nn.Linear(100,10))

model.to(device)

#Importing the data
train_data = datasets.CIFAR10(root='./datasets',
                              transform=transforms.ToTensor(),
                              download=True,
                              train=True)
test_data = datasets.CIFAR10(root='./datasets',
                             transform=transforms.ToTensor(),
                             download=True,
                             train=False)

#Splitting entire dataset into batches using dataloader
train_loader = DataLoader( batch_size=batch_size,
                           shuffle=True,
                           dataset=train_data)
test_loader = DataLoader( dataset=test_data,
                          batch_size=batch_size,
                          shuffle=True)

# Writing the training and test function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( model.parameters(), lr = learning_rate)

def model_train():
# training the model on train_loader
    for epoch in range(num_epochs):
        for idx, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            model.to(device)
            loss = loss_fn(model(img), label).to(device)

            model.train()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# check accuracy of the model
def model_accuracy():

    with torch.no_grad():

        samples = 0
        correct = 0

        for img, label in test_loader:
            img, label = img.to(device), label.to(device)

            _ , pred_out = model(img).max(1)
            correct += (pred_out == label).sum().item()
            samples += label.size(0)

        acc = correct / samples
        print("--------------------------------------------")
        print(f"Accuracy == {acc*100}%")
        print("--------------------------------------------")

model_train()
model_accuracy()


































