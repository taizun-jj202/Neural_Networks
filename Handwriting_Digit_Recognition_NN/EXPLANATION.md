### Digit Recognition using Linear Neural Network
----
This is a simple neural network for classifying handwriting digits from the MNIST dataset that uses 3 hidden layers.

I have chosen 3 hidden layers for the purpose of experimentation to see what would happen if i increase hidden layer count. 
Initially used a single hidden layer, but quickly found out that the accuracy of the network was very poor at ~10%.


```python
#Definig the different parameters
num_epochs = 10
num_classes = 10 #Multi-class clasification (i.e there are 10 digits)
learning_rate = 0.001
batch_size = 50
input_size = 784 #Because we flatten a 28*28 image into a single row and feed the row as input
hidden_layers = 100
```
Defining all the parameters that might be used in multiple cells together so that  changing them later would not be a problem


```python
train_data = torchvision.datasets.MNIST(root = "./dataset", train = True, transform = transforms.ToTensor(), download = True)
test_data = torchvision.datasets.MNIST(root = "./dataset", train = False, transform = transforms.ToTensor(), download = True)
print("Size of training data: {}".format(len(train_data)))
print("Size of testing data : {}".format(len(test_data)))

#Below lines give us the samples for training and testing
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False, num_workers=2)
```

``train_data`` and `test_data` variables are used to import training and testing datasetst respectively from the MNIST dataset from PyTorch. Since PyTorch library works using tensor(a sort of array), we need to convert each sample that we import from the training dataset into a tensor ( `.ToTensor()`)

`train_loader` variable is used to create batches to train the NN on, and also make it easier to iterate over each sample
`test_loader` is the same as train_loader but used for test data.

```python
checkdata = iter(train_loader)
img, lab = next(checkdata)
print(img.shape, lab.shape)
```
`checkdata` is a iterative variable that iterates over the `train_loader` defined earlier and returns a sample every iteration. 



#### Neural Network for the problem:
---
```python
	
class digit_recon(nn.Module):

  def __init__(self, input_size, hidden_layers, num_classes):
    super(digit_recon, self).__init__()
    #First layer is given below
    self.input = nn.Linear(in_features = input_size, out_features = hidden_layers)
    self.relu_1 = nn.ReLU() #Invoking the activation function
    self.hidden_1 = nn.Linear(in_features = hidden_layers, out_features = hidden_layers)
    self.relu_2 = nn.ReLU()
    self.hidden_2 = nn.Linear(in_features = hidden_layers, out_features = hidden_layers)
    self.relu_3 = nn.ReLU()
    self.hidden_3 = nn.Linear(in_features = hidden_layers, out_features = hidden_layers)
    self.relu_4 = nn.ReLU() 
    self.output = nn.Linear(in_features = hidden_layers, out_features = num_classes)

  def forward(self,x):  
    model = self.input(x)
    model = self.relu_1(model)
    model = self.hidden_1(model)
    model = self.relu_2(model)
    model = self.hidden_2(model)
    model = self.relu_3(model)
    model = self.hidden_3(model)
    model = self.relu_4(model)
    model = self.output(model)

    return model
```

We define the neural network as a class that inherits from nn.Module in PyTorch. 
`__init()__` function is a constructor that takes 3 parameters as input: number of inputs to the NN, number of hidden layers, number of output nodes.
super() function is used to call `__init()__` of the parent class(i.e the NN class)
We do this to perform any neccessay initializations the parent class requires.

Then the next few lines define what functions we need to set the weights and biases for the hidden layers as well as what needs to be the output 

`def forward(self,x)` function defines the forward pass of the NN model. It describes how data will flow through the layers of the Network


```python
model = digit_recon(input_size, hidden_layers, num_classes)
repr(model)
```

This line of code shows us the shape of the NN so we cna check if all layers appear proper or not.

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
```
`criterion` is the loss function applied to this NN and we use the `Adam()` optimizer to minimize the loss function for the best accuracy possible in this NN. We can use other optimizers as well.


#### Training Loop:
```python
for epoch in range(num_epochs):
  #Loop for batch training
  for (images, labels) in train_loader:
    images = images.reshape(-1,784).to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    #Backpropagating
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```
1. Epochs are the number of passes the NN makes over the entire dataset. We use multiples epochs so that the NN can make multiple passes over the dataset to improve its ability to generalize.
2. Then we pass individual datapoints into the NN so that it can set all weights and biases in the first pass.
3. We repeat the number of passes to improve the initially set weights and biases.
4. Then we add backpropagation to the NN


### Finding the Accuracy of the Model: 
---
Now we use the followng code to find accuracy of the above created NN:

```python
#Finding the accuracy of the model

def model_test():
    with torch.no_grad(): #Disables gradient calculation
      correct = 0
      tot_samples = 0

      for (images, lab) in test_loader:
        images = images.reshape(-1,784).to(device)
        lab = lab.to(device)

        #Getting the highest predicted value for the classes
        #Highest probability given to the class that model predicts is correct
        outputs = model(images)
        _ , predicted = torch.max(outputs,1)
        tot_samples += len(lab)
        correct += (predicted == lab).sum().item()

    accuracy = 100 * (correct / tot_samples)
    print("Accuracy : {}".format(accuracy))

model_test()
```

Since we do not want to update the gradient as we are finding the accuracy, we use the `no_grad()` function.
Similar to the training loop, we iterate over the dataset having images and corresponding labels but we skip the part on updating the loss function.

To find accuracy, we simply divide the samples NN identified correctly  by the Total number of samples fed into it.

`torch.max()` returns two values, one  is the identified image(wheather incorrect or correct) and the second value is the label corresponding to the image.
Since we only require the label to identify the correct image, we discard the first argument returned. 

Then we sum the correctly identfied labels and total labels passed(as 1label passed into the model means that an images also has to be passed) individually, and use these two values to figure out the accuracy.

The code from below this point onwards is just so that i can save and load the calculated weights to my Google Drive and not have to train the model every time i quit and rejoined the Colab Runtime.
