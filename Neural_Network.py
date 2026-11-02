#This will be for neural network
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X,y = load_breast_cancer(return_X_y=True)
#This shows us a numpy array about a tumor where y is 0 or 1 if tumor is malignant or not

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#This means we are using 20% of data for evaluation and 80% for testing

#Neural Networks are scale sensitive so we need to scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled) #numpy array of scaled data
print(y_train) #a bunch of 0s and 1s indicating cancerous or not

X_train_scaled_tensor = torch.from_numpy(X_train_scaled).float()
X_test_scaled_tensor = torch.from_numpy(X_test_scaled).float()
y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)
#Create tensor datasets

train_dataset = TensorDataset(X_train_scaled_tensor, y_train_tensor)

X_train_scaled_tensor.shape

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#Load max data set of 32 and shuffle it for randomness

###Create Neural Network
class BCNet(nn.Module):
    def __init__(self):
        super(BCNet, self).__init__() #Construct network
        
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        #A neural network is called that bc it mimics the brain neurons
        #So each fully connected layer (fc) ends and begins with the previous like a connected network

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x

model = BCNet()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

###Train neural network

epochs = 20 #An epoch is 1 complete pass of an entire training dataset

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for x_batch,y_batch in train_loader:
        optimizer.zero_grad()

        preds = model(x_batch)
        loss = criterion(preds, y_batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss was {running_loss / len(train_loader)}")
