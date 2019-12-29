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
    self.mu_layer = nn.Linear(400, 2)
    self.logvar_layer = nn.Linear(400, 2)
    self.decoder = nn.Sequential(
      nn.Linear(2, 400),
      nn.ReLU(True),
      nn.Linear(400, 784),
      nn.Tanh()
    )

  def reparam_(self, mu, logvar):
    std = 0.5 * torch.exp(logvar)
    epsilon = torch.rand_like(std)

    return mu + std * epsilon

  def encode(self, x):
    x = self.encoder(x)
    mu, logvar = self.mu_layer(x), self.logvar_layer(x)
    return mu, logvar

  def decode(self, x):
    return self.decoder(x)

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparam_(mu, logvar)
    x = self.decode(z)
    return x

ae = MNISTAE().cuda()
ae.load_state_dict(torch.load("mnist_basic_vae_weights.pth"))

data = []
data_mu = []
data_logvar = []
targets = []

n_samples = int(len(trainfolder) * 0.25)
counter = 0

for batch_x, batch_y in tqdm(trainloader):
  batch_x = batch_x.cuda().float().view(-1, 784)
  
  mu, logvar = ae.encode(batch_x)

  z = ae.reparam_(mu, logvar).detach().cpu().numpy()

  for x, y in zip(z, batch_y):
    data.append(x.reshape(2))
    targets.append(y)
    counter += 1

  if counter >= n_samples:
    break

data = np.array(data)
targets = np.array(targets)

df = pd.DataFrame({"x":data[:, 0], "y":data[:,1], "hue":targets})
sns.scatterplot(x="x", y="y", hue="hue", data=df)
plt.savefig("visualizations/mnist_basic_vae_data_visualization.png")
