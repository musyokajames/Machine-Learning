import plistlib
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#Create known parameters
weight = 0.7
bias = 0.3

#create some data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim = 1)
y = weight * X + bias

#splitting the data into training and test datasets
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

#plot predictions
def plot_predictions(train_data = X_train,
                     train_labels =y_train,
                     test_data = X_test,
                     test_labels = y_test,
                     predictions = None):
    
    plt.figure(figsize = (10, 7))

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")

    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label ="Predictions")

    plt.legend(prop = {"size" : 14});
    plt.show()
# plot_predictions()

#Build the model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(in_features = 1,
                                      out_features = 1)
        
    #Forward pass
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    

#Check contents of the model
torch.manual_seed(42)   
model_1 = LinearRegressionModel()
# print(list(model_1.parameters()))

#Train the model
#Set up a loss function
loss_fn = nn.L1Loss()

#Set up an optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr = 0.01)

#Set up a training loop
torch.manual_seed(42)
epochs = 200

#optimization song
    # It's train time!
    #for epoch in a range
    # do the forward pass
    # calculate the loss
    # optimizer zero grad
    # lossss backward
    # optimizer step step step

    # let's test now
    #in model.eval
    # with torch.inference mode
    # do the forward pass 
    # calculate the loss
    # watch it go down down down

for epoch in range(epochs):

    model_1.train()

    y_pred = model_1(X_train)

    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    #Testing loop
    model_1.eval()

    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"Epoch : {epoch} | Loss : {loss} | Test loss : {test_loss}")

#Make predictions with the model
with torch.inference_mode():
    y_preds = model_1(X_test)
    plot_predictions(predictions=y_preds)
