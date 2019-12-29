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
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.5,), std=(0.5,))
])

trainfolder = datasets.ImageFolder(root=TRAINING_DIR, transform=trans)
trainloader = data.DataLoader(trainfolder, batch_size=128, num_workers=6, shuffle=True)

class MNISTAE(nn.Module):
  def __init__(self):
    super().__init__()

    self.encoder = nn.Sequential(
      nn.Linear(784, 400),
      nn.ReLU(True)
    )
    self.decoder = nn.Sequential(
      nn.Linear(400, 784),
      nn.Tanh()
    )

  def encode(self, x):
    return self.encoder(x)

  def decode(self, x):
    return self.decoder(x)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

ae = MNISTAE().cuda()
ae.load_state_dict(torch.load("mnist_basic_autoencoder_weights.pth"))

data = []
targets = []

n_samples = int(len(trainfolder) * 0.25)
counter = 0

for batch_x, batch_y in tqdm(trainloader):
  batch_x = batch_x.cuda().float().view(-1, 784)
  batch_x_preds = ae.encode(batch_x).detach().cpu().numpy()
  batch_y = batch_y.detach().cpu().numpy()

  for x, y in zip(batch_x_preds, batch_y):
    data.append(x.reshape(400))
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
plt.savefig("visualizations/mnist_basic_autoencoder_data_visualization.png")
