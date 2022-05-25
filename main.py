import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader

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

print("torch", torch.__version__)
print("device", device)
print(torch.backends.mps.is_available())



#X_all = torch.Tensor(np.random.randn(100, 200,10)).to(device)
#y_all = torch.Tensor(np.random.randn(100, 200, 1)).to(device)
X= torch.randn(10000, 200, 10, device=device)
y = torch.randn(10000, 200, 1, device=device)


# Train
#model = NN_mps()
model = NN().to(device)
loss_f = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
model.train()

for it in range(EPOCHS):



    #list_3dim = []
    for i in range(20):
        #X, y = X_all[:,i,:], y_all[:,i,:]
        #X, y = X_all, y_all
        y_pred = model(X)
        #list_3dim.append(model.encoder(model.flatten(X)))
        
        loss = loss_f(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #time.sleep(1)


#print(loss.mean())

