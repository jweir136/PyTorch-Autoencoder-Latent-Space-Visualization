import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

TRAINING_DIR = "../../datasets/mnist-jpg/trainingSet/trainingSet"
trans = transforms.Compose([
  transforms.Grayscale(),
  transforms.Resize(28),
  transforms.ToTensor()
])

trainfolder = datasets.ImageFolder(root=TRAINING_DIR, transform=trans)
trainloader = data.DataLoader(trainfolder, batch_size=128, num_workers=6, shuffle=True)

data = []
targets = []

n_samples = int(len(trainfolder) * 0.55)
counter = 0

for batch_x, batch_y in tqdm(trainloader):
  batch_x = batch_x.detach().cpu().numpy()
  batch_y = batch_y.detach().cpu().numpy()

  for x, y in zip(batch_x, batch_y):
    data.append(x.reshape(784))
    targets.append(y)
    counter += 1

  if counter >= n_samples:
    break

data = np.array(data)
targets = np.array(targets)

data = data[:int(len(data)*0.25)]
targets = targets[:int(len(targets)*0.25)]

data = TSNE(n_components=2, perplexity=15, learning_rate=10, verbose=2).fit_transform(data)

df = pd.DataFrame({"x":data[:, 0], "y":data[:, 1], "hue":targets})

sns.scatterplot(x="x", y="y", data=df, hue="hue", legend="full")
plt.savefig("visualizations/mnist_data_visualization.png")
