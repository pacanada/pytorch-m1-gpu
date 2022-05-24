import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

m1_gpu = torch.device("mps")
m1_cpu = torch.device("cpu")
EPOCHS = 10
GPU = True

if GPU == True:
    device = m1_gpu
else:
    device = m1_cpu

X_all = torch.Tensor(np.random.randn(10, 20, 10)).to(device)
y_all = torch.Tensor(np.random.randn(10, 20, 1)).to(device)


# Train
model = NN().to(device)
loss_f = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
model.train()

for it in range(EPOCHS):
    #list_3dim = []
    for i in range(20):
        X, y = X_all[:,i,:], y_all[:,i,:]
        y_pred = model(X)
        #list_3dim.append(model.encoder(model.flatten(X)))
        
        loss = loss_f(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


print(loss.mean())

