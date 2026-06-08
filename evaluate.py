import matplotlib.pyplot as plt
from model import *
from train import *
from dataset import *
for batch_idx, (data, labels) in enumerate(test_loader):
    data = data.to(device)
    optimizer.zero_grad()
    data_noise = torch.randn(data.shape).to(device)
    data_noise += data
    recon_batch = model(data_noise)

plt.figure(figsize = (20, 12))
for i in range(5):
    print(f"Image {i} with label {labels[i]}", end = "")
    plt.subplot(3, 5, 1 + i)
    plt.imshow(data_noise[i, :, :, :].view(28, 28).cpu().detach().numpy(), cmap = 'binary')
    plt.axis('off')
    plt.subplot(3, 5, 6 + i)
    plt.imshow(recon_batch[i, :].view(28, 28).cpu().detach().numpy(), cmap = 'binary')
    plt.axis('off')
    plt.subplot(3, 5, 11 + i)
    plt.imshow(data[i, :, :, :].view(28, 28).cpu().detach().numpy(), cmap = 'binary')
plt.show()