from tkinter import Y
from sympy import plot
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as ps
import numpy as np
import sklearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary

#Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

#Create multi-class data
X_blob, y_blob = make_blobs(n_samples = 2000, #type:ignore
                            n_features = NUM_FEATURES,
                            centers = NUM_CLASSES,
                            cluster_std = 1.5,
                            random_state = RANDOM_SEED)

#Turn Data to tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.long)

#Split into train and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size = 0.2,
                                                                        random_state=42)

#Plot
plt.figure(figsize=(12, 6))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap= plt.cm.RdYlBu) #type:ignore
# plt.show()

#Build a classification model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units = 8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

#Create instance of the model
model_1 = BlobModel(input_features=2,
                    output_features=4,
                    hidden_units=8)

# print(model_1)
#Create loss fn and optimizer and accuracy
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params = model_1.parameters(),
                            lr = 0.01)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred) * 100)
    return acc

#Create training and test loop
torch.manual_seed(42)

epochs = 2000

#Training Loop
for epoch in range(epochs):
    model_1.train()

    #Forward pass
    y_logits = model_1(X_blob_train)
    y_pred = torch.softmax(y_logits , dim=1).argmax(dim=1)

    #Calculate the loss and accuracy
    loss = loss_fn(y_logits, y_blob_train)

    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)
    
    #Optimizer zero grad
    optimizer.zero_grad()

    #Back propagation
    loss.backward()

    #Gradient Descent
    optimizer.step()

    #Testing loop
    model_1.eval()
    with torch.inference_mode():
        #Forward pass
        test_logits = model_1(X_blob_test)
        test_pred = torch.softmax(test_logits, dim = 1).argmax(dim=1)

        #Calcualte the loss and accuracy
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_pred=test_pred)
        
        #Print out what's happening
        if epoch % 100 == 0:
            print(f"Epoch : {epoch} |Loss : {loss:.4f},Acc : {acc:.2f}% | Test loss : {test_loss:.4f}, Test Acc : {test_acc:.2f}%")

#Plot
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_blob_test, y_blob_test)
plt.show()