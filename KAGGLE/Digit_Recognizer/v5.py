import torch
import torch.nn as nn
import pandas as pd
from CNN_MODEL import dig_rec,custom_call # The CNN module
from torch.utils.data import random_split,DataLoader,Dataset

# Setting device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device in use : {device}")

# Reading csv files and splitting them
test_csv  = pd.read_csv('test.csv')
train_csv = pd.read_csv('train.csv')
print(f"Length of train.csv : {len(train_csv)}")
print(f"Length of test.csv  : {len(test_csv)}")

tr_img_df = train_csv.iloc[:, 1:] # All cols except [0]
tr_label_df = train_csv.iloc[:,0] # First col only

tst_img_df = test_csv.iloc[:]

# declaring our model
model = dig_rec()
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Creating batches
train_set = custom_call(dataset_csv=train_csv,
                        label_df = tr_label_df,
                        img_df   = tr_img_df)

train_tr_split, train_tst_split = random_split(train_set, [33600, 8400])

train_data = DataLoader(dataset=train_tr_split,
                        batch_size=24,
                        shuffle=True)
test_data = DataLoader(dataset=train_tst_split,
                       shuffle=True,
                       batch_size=24)

# Training the model
num_epochs = 5
def train():

    print("...Initializing training...")

    for epochs in range(num_epochs):
        print(f"Epoch: {epochs}/{num_epochs}")
        for idx,(label, img) in enumerate(train_data):
            label, img = label.to(device), img.to(device)

            img = img.reshape(-1,1,28,28)

            predicted_label = model(img)
            loss = loss_fn(predicted_label, label)
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
    print("...Finished training... ")


train()

def model_acc():

    tot_correct = 0
    tot_samples = 0
    print("...Calculating Accuracy of the model")
    for (label,img) in range(test_data):
        label, img = label.to(device), img.to(device)

        img = img.reshape(-1,1,28,28)
        predictions = model(img)
        tot_correct += (predictions == label).sum().item()
        tot_samples += label.size(0)

    acc = ((tot_correct/tot_samples) * 100)
    print("Accuracy :")



# import torchvision # Serves as a stop for debugger only

































