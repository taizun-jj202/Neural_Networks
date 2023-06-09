import torch
import pandas as pd
from torch.utils.data import random_split,DataLoader,Dataset
from MODEL import *

train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device in Use: {device}")
print(f"Length of train.csv : {len(train_csv)}")
print(f"Length of test.csv  : {len(test_csv)}")

# Extracting data and loading data into batches for training
train_data = custom_call(train_csv)
train_loader = DataLoader(dataset=train_data,
                          batch_size=24,
                          shuffle=True)

# Declaring model, lossfnc, optimizer
model = nnmodel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# training , accuracy, saving , loading loop
num_epochs = 5
def train():
    print("... Training model ...")
    for epoch in range(num_epochs):
        print(f"Epoch : {epoch}/ {num_epochs}")
        for i,l in train_loader:
            i, l = i.to(device), l.to(device)

            output = model(i)
            optimizer.zero_grad()
            loss = loss_fn(output, l)
            loss.backward()
            optimizer.step()
    print("... Finished training ...")

def acc():
    tot_correct = 0
    tot_samples = 0
    print("Calculating accuracy of the model....")
    print("Calculating accuracy on 'train.csv' file ")
    for i,la in train_loader:
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

train()
save()
acc()
load()


# Loop for test.csv file now
outputs = []
print("... Calculating indices of test_csv and appending to a list ...")
for i in range(len(test_csv)):
    data = torch.tensor(test_csv.iloc[i]).to(device)
    out = model(data)
    prediction = out.argmax(0).tolist()
    outputs.append(prediction)

print(f"Shape of outputs: {len(outputs)}")
print("Creating Submission.csv file")
sub = pd.read_csv('sample_submission.csv')
sub['Label'] = outputs
print("Appending predicted outputs to file ...")
sub.to_csv('test1.csv', index=False)