import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from torch import nn, optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor()#,
    # transforms.Normalize(0, 1)
])

mnist_dataset_train = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
mnist_dataset_test = datasets.MNIST(root = './data', train = False, download = True, transform = transform)


train_loader = torch.utils.data.DataLoader(mnist_dataset_train, batch_size = 128, shuffle = True)

test_loader = torch.utils.data.DataLoader(mnist_dataset_test, batch_size = 128, shuffle = False)

