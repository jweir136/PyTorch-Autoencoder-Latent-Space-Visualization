import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import sys
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

class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size()[0], -1)

class UnFlatten(nn.Module):
  def forward(self, x):
    return x.view(x.size()[0], 16, 20, 20)

class MNISTAE(nn.Module):
  def __init__(self):
    super().__init__()

    self.encoder = nn.Sequential(
      nn.Conv2d(1, 8, kernel_size=5),
      nn.ReLU(True),
      nn.Conv2d(8, 16, kernel_size=5),
      nn.ReLU(True),
      Flatten(),
      nn.Linear(6400, 400),
      nn.ReLU(True)
    )
    self.mu_layer = nn.Linear(400, 2)
    self.logvar_layer = nn.Linear(400, 2)
    self.decoder = nn.Sequential(
      nn.Linear(2, 400),
      nn.ReLU(True),
      nn.Linear(400, 6400),
      nn.ReLU(True),
      UnFlatten(),
      nn.ConvTranspose2d(16, 8, kernel_size=5),
      nn.ReLU(True),
      nn.ConvTranspose2d(8, 1, kernel_size=5),
      nn.Tanh()
    )

  def reparam_(self, mu, logvar):
    std = torch.exp(logvar)
    epsilon = torch.rand_like(std)
    return mu + std * epsilon

  def encode(self, x):
    x = self.encoder(x)
    mu, logvar = self.mu_layer(x), self.logvar_layer(x)
    return mu, logvar

  """
    [!] Other methods have been ommited.
  """
    
ae = MNISTAE().cuda()
ae.load_state_dict(torch.load("mnist_conv_vae_weights.pth"))

data = []
data_mu = []
data_logvar = []
targets = []

n_samples = int(len(trainfolder) * 0.25)
counter = 0

for batch_x, batch_y in tqdm(trainloader):
  batch_x = batch_x.cuda().float()
  
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
plt.savefig("visualizations/mnist_conv_vae_data_visualization.png")
