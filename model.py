import torch
import torch.utils.data
from torch import nn, optim
import pandas as pd


noise_factor = 0.5
criterion = nn.MSELoss()


class DAE(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))
    
    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))
    
    def forward(self, x):
        q = self.encode(x.view(-1, 784))
        return self.decode(q)

def train(epoch, model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate (train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        # data_noise = data + noise_factor * torch.randn_like(data)
        data_noise = torch.randn(data.shape).to(device)
        data_noise += data
        # torch.clamp(data_noise, 0, 1)
        recon_batch = model(data_noise)
        loss = criterion(recon_batch, data.view(data.size(0), -1))
        loss.backward()
        train_loss += loss.item() * len(data)
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{}({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx/len(train_loader), loss.item()))
    print('==> Epoch: {} Average Loss: {:.4f}'.format(epoch, train_loss/ len(train_loader.dataset))) 
    print(data_noise.mean(), '\n', data_noise.std())


