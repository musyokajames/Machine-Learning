import torch
from torch import nn 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torchmetrics
from torchmetrics import Accuracy
from helper_functions import plot_decision_boundary

#Set up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLES = 1000
RANDOM_SEED = 42

#Create a dataset
X_moon, y_moon = make_moons(n_samples = NUM_SAMPLES,
                            noise = 0.07,
                            random_state = RANDOM_SEED)

#Turn data into a DataFrame
moons = pd.DataFrame({"X1" : X_moon[:, 0],
                      "X2" : X_moon[:, 1],
                      "label" : y_moon})
# print(moons.head(10))

#Visualize the data on a scatter
plt.scatter(X_moon[:, 0], X_moon[: ,1], c=y_moon, cmap = plt.cm.RdYlBu) #type:ignore
# plt.show()

#Turn data into Tensors
X_moon = torch.tensor(X_moon, dtype=torch.float)
y_moon = torch.tensor(y_moon, dtype=torch.float)

#Split into train and test sets
X_moon_train, X_moon_test, y_moon_train, y_moon_test = train_test_split(X_moon,
                                                                        y_moon,
                                                                        test_size=0.2,
                                                                        random_state=42)

#Build a model
class MoonModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units ):
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

#Instantiate the model
model_1 = MoonModel(input_features=2,
                    output_features=1,
                    hidden_units=10).to(device)

#Setup Loss Function
loss_fn = nn.BCEWithLogitsLoss()

#Setup optimizer to optimize model's parameters
optimizer = torch.optim.SGD(params = model_1.parameters(),
                            lr=0.1)

#Setup Accuracy
acc_fn = Accuracy(task="multiclass", num_classes = 2).to(device)

#Create a training and testing loop
torch.manual_seed(RANDOM_SEED)

#Setup epochs
epochs = 2000

#Send data to device
X_moon_train, y_moon_train = X_moon_train.to(device), y_moon_train.to(device)
X_moon_test, y_moon_test = X_moon_test.to(device), y_moon_test.to(device)

#Training
#Loop through the data
for epoch in range(epochs):
    model_1.train()

    #Forward pass
    y_logits = model_1(X_moon_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    #Calculate the loss and accuracy
    loss = loss_fn(y_logits, y_moon_train)

    acc = acc_fn(y_pred, y_moon_train.int())

    #Zero the gradients
    optimizer.zero_grad()

    #Loss backward(perform back propagation)
    loss.backward()

    #Step the optimizer(Gradient Descent)
    optimizer.step()

    #Testing
    model_1.eval()
    with torch.inference_mode():
        #Forward pass
        test_logits = model_1(X_moon_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        #Calculate the loss/acc
        test_loss = loss_fn(test_logits, y_moon_test)
        test_acc = acc_fn(test_pred, y_moon_test.int())

    #Print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch:{epoch} | Loss:{loss:.2f}, Acc:{acc:.2f}% | Test loss :{test_loss:.2f}, Test Acc: {test_acc:.2f}%")

#Plot decision boundaries for training and test sets
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_moon_train, y_moon_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_moon_test, y_moon_test)
plt.show()

