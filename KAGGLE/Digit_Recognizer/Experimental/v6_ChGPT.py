import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.model_selection import train_test_split
from CNN_MODEL import dig_rec, custom_call

# Setting device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device in use: {device}")

# Reading csv files
train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')

# Splitting into train and validation sets
train_data, val_data = train_test_split(train_csv, test_size=0.2, random_state=42)

# Declaring our model
batch_size = 24
model = dig_rec(batch_size)
model.to(device)

# Data preprocessing and augmentation
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])  # Normalize the pixel values to the range [-1, 1]
])

# Creating datasets and data loaders
train_set = custom_call(dataset_csv=train_data, transform=transform)
val_set = custom_call(dataset_csv=val_data, transform=transform)
test_set = custom_call(dataset_csv=test_csv, transform=transform)

train_data = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_data = DataLoader(dataset=val_set, batch_size=batch_size)
test_data = DataLoader(dataset=test_set, batch_size=batch_size)

# Training the model
num_epochs = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    print("...Initializing training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for labels, images in train_data:
            labels, images = labels.to(device), images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_set)
        print(f"Epoch: {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for labels, images in val_data:
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

# Training the model
train()
# Evaluate the model on validation data
evaluate()
# Save the trained model
torch.save(model.state_dict(), 'model_weights.pth')
