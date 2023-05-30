# 1) Design the model(input, output, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#     - forward pass : compute predictions
#     - backward pass: calculate gradients
#     - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets # Import binary classification dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Prepare data
bc = datasets.load_breast_cancer()
x,y = bc.data, bc.target

n_samples, n_features = x.shape
xtrain, xtest , ytrain, ytest = train_test_split(x,y , test_size=0.2, random_state=1234)

# scale the features
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

xtrain = torch.from_numpy(xtrain.astype(np.float32))
xtest = torch.from_numpy(xtest.astype(np.float32))
ytrain = torch.from_numpy(ytrain.astype(np.float32))
ytest = torch.from_numpy(ytest.astype(np.float32))

ytrain = ytrain.view(ytrain.shape[0], 1)
ytest = ytest.view(ytest.shape[0], 1)

# 1) model
# f = wx + b, sigmoid at the end
class Logrgr(nn.Module):
        def __init__(self, input_features):
            super(Logrgr, self).__init__()
            self.linear = nn.Linear(input_features, 1)

        def forward(self,x):
            y_predicted = torch.sigmoid(self.linear(x))
            return y_predicted


model = Logrgr(n_features)

# 2) loss and optimizer
learning_rate = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 3) training loop
n_epochs = 80000

    # training loop
for epoch in range(n_epochs):
    # forward pass
    y_predicted = model(xtrain)
    loss = criterion(y_predicted, ytrain)
    # backward pass
    loss.backward()
    # updating weights
    optimizer.step()
    optimizer.zero_grad() # emptying the gradients for back-pass

    if (epoch+1) % 100 == 0:
        print(f"Epoch: {epoch +1}, Loss = {loss.item():.4f}")
    if loss == 0:
        exit(0)

# evaluating the model now

with torch.no_grad():
    y_predicted = model(xtest)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(ytest).sum() / float(ytest.shape[0])
    print(f"Accuracy = {acc:.4f}")