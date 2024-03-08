import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

#Code for creating a spiral dataset
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

N = 100 #number of points per class
D = 2 #Dimensionality
K = 3 #number of classes
X = np.zeros((N*K, D)) #data matrix (each row = single example)
y = np.zeros(N*K, dtype="uint8") #class labels
for j in range(K):
    ix = range(N*j, N*(j + 1))
    r = np.linspace(0.0, 1, N) #radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 #theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

#Let's visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu) #type:ignore
# plt.show()

#Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.long)

#Create train and test splits
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=RANDOM_SEED)

#Let's calculate the accuracy for when we fit our model
import torchmetrics
from torchmetrics import Accuracy
acc_fn = Accuracy(task="multiclass", num_classes = 4).to(device)

#Create a model
class SpiralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=10)
        self.linear2 = nn.Linear(in_features=10, out_features=10)
        self.linear3 = nn.Linear(in_features=10, out_features=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear3(self.relu(self.linear2(self.relu(self.linear1(x)))))

#Instantiate model and send it to device   
model_1 = SpiralModel()

#Setup data to be device agnostic
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

#Print out first 10 untrained model outputs(foward pass)
# print("Logits:")
# print(model_1(X_train)[:10])

# print("Pred probs:")
# print(torch.softmax(model_1(X_train)[:10], dim = 1))

# print("Pred labels:")
# print(torch.softmax(model_1(X_train)[:10], dim = 1).argmax(dim=1))

#Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params = model_1.parameters(),
                             lr = 0.1)

#Build a training loop for the model
torch.manual_seed(RANDOM_SEED)
epochs = 2000

#loop through the data
for epoch in range(epochs):
    #Training
    model_1.train()

    #forward pass
    y_logits = model_1(X_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    #calculate the loss and acc
    loss = loss_fn(y_logits, y_train)
    acc = acc_fn(y_pred, y_train)

    #Optimizer zero grad
    optimizer.zero_grad()

    #loss backward
    loss.backward()

    #optimizer step
    optimizer.step()

    #Testing
    model_1.eval()
    with torch.inference_mode():
        #forward pass
        test_logits = model_1(X_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

        #Calculate the loss
        test_loss = loss_fn(test_logits, y_test)
        test_acc = acc_fn(test_pred, y_test)

    #print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch:{epoch} | Loss:{loss:.2f}, Acc:{acc:.2f}% | Test loss :{test_loss:.2f}, Test Acc: {test_acc:.2f}%")

#Plot decision boundary for training and test sets
from helper_functions import plot_decision_boundary
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
plt.show()
