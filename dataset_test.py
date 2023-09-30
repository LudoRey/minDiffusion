from datasets.axons_dataset import AxonsDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt

dataset = AxonsDataset(root="./data", phase="test", crop=False, normalize=False)
dataloader = DataLoader(dataset, batch_size=60)

data = next(iter(dataloader))['confocal']

data = data.cpu().numpy()
data_flat = data.flatten()

hist, bins = np.histogram(data_flat, bins=256, range=(0, 256))

# Plot the histogram
plt.figure(figsize=(8, 6))
plt.hist(data_flat, bins=256, range=(0, 1), color='b', alpha=0.7)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of dataset')
plt.grid(axis='y', alpha=0.75)
plt.savefig('histogram.png', dpi=200)


