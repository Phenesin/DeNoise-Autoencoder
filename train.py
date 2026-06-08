from dataset import device, train_loader
from model import DAE, train
from torch import optim


epochs = 10
model = DAE().to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

for epoch in range(1, epochs + 1):
    train(epoch, model, train_loader, optimizer, device)

