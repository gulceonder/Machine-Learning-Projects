#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 19:21:03 2022

@author: gulceonder
"""
import torch
from torch import nn
from torch.utils.data import DataLoader , random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import numpy as np
from matplotlib import pyplot as plt
import pickle

# Download training data from open datasets.6000 samples
training_data = datasets.FashionMNIST(
    root="data", #path where data is stored
    train=True,  #train or testa
    download=True, #download if not in root
    transform=ToTensor(),#normalize pixel values betwenn [0,1]
    
)

# Download test data from open datasets.10000 samples
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

unused, valid = random_split(training_data,[50000,10000])
batch_size = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
valid_dataloader = DataLoader(valid, batch_size=10000)
val_size = len(valid_dataloader.dataset)
test_size = len(test_dataloader.dataset)
train_size = len(train_dataloader.dataset)

#see the shape of loaded data
# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# mlp_2 classifier
class mlp_2(torch.nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2, num_classes):
        super(mlp_2, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size2, num_classes,bias=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = x.view(-1, self.input_size)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.fc3(output)
        return output
# initialize your model
model = mlp_2(784,16,64,10).to(device)


print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

loss_data=np.zeros((10,15, 94))#store training loss matrices 
train_acc_data= np.zeros((10,15,94)) #store training accuracy matrices
valid_acc_data= np.zeros((10,15,94)) #store validation accuracy matrices
test_acc_data= np.zeros((10)) #store test accuracy values
weights_data =np.zeros((10,16,784))#weights data for mlp_1

#reset parameterds after eaach run code from: https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
        
        
        
def train( model, loss_fn, optimizer,epoch,run):


    model.train()


    for batch, (X, y) in enumerate(train_dataloader):
        correct= 0
        correct_val=0
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if batch % 10 == 0:
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss, current = loss.item(), batch 
            train_acc =100*(correct/batch_size) #training accuracy for the given batch
            #store the loss and accuracy in numpy arrays
            curr_batch=int((current/10))

            loss_data[run][epoch][curr_batch]=loss
            train_acc_data[run][epoch][curr_batch]=train_acc

            #calculate validation accuracy


            model.eval()
            with torch.no_grad():
                for X, y in valid_dataloader:
                    X, y = X.to(device), y.to(device)
                    pred2 = model(X)
                    correct_val += (pred2.argmax(1) == y).type(torch.float).sum().item()

            val_acc = 100*(correct_val/val_size) #training accuracy up to the current batch 
            valid_acc_data[run][epoch][curr_batch]=val_acc
            print(f"loss: {loss:>7f} training accuracy: {train_acc:>0.1f}% , validation accuracy= {val_acc:>0.1f}%  [{current:>5d}/930]")
            
def test( model, loss_fn,run):

    # num_batches = len(test_dataloader)
    
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # test_loss /= num_batches
    correct /= test_size
    # get the parameters 784x128 layer as numpy array
    weights_data[run]= model.fc1.weight.data.numpy()
    test_acc_data[run]= 100*correct
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}% \n")

  
runs=1#number of times to run program
for run in range(runs):  
    epochs = 1 #train and test data for 15 epochs 
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train( model, loss_fn, optimizer,t,run)
    test( model, loss_fn,run)#test 1 one time for each run
    model.apply(weight_reset)
    print("Done!")
    


# print (train_acc_data)
train_acc_data=train_acc_data.reshape(10,1410);#linearize curves
# print(train_acc_data.shape)
train_acc_data=train_acc_data.mean(0) #take average of curves for 10 runs 
# print(train_acc_data.shape)
# print(train_acc_data)

valid_acc_data=valid_acc_data.reshape(10,1410);#linearize curves
valid_acc_data=valid_acc_data.mean(0) #take average of curves for 10 runs 


#average 
loss_data=loss_data.reshape(10,1410);#linearize curves
loss_data=loss_data.mean(0) #take average of curves for 10 runs 

max_acc = np.max(test_acc_data) #best test accuracy from all runs
max_acc_index=np.argmax(test_acc_data)


max_acc_weights=weights_data[max_acc_index] #weights of the first layer of best test accuracy

# Creating a Dictionary
Dict = {'name': 'mlp_1', 'loss_curve': loss_data, 'train_acc_curve': train_acc_data, 'val_acc_curve': valid_acc_data, 'test_acc_data': max_acc, 'weights': max_acc_weights}

#save it using pickle
file_to_write = open("output.pkl", "wb")
pickle.dump(Dict, file_to_write)


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
