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

  def decode(self, x):
    return self.decoder(x)

  """
    [!] Other methods have been ommited.
  """
    
ae = MNISTAE().cuda()
ae.load_state_dict(torch.load("mnist_conv_vae_weights.pth"))

df = pd.read_csv("mnist_conv_vae_dataframe.csv")

######################### FIND THE MEAN AND STD FOR THE VALUES IN THE LATENT SPACE THAT MAKE UP 1 AND 0 ##########################

mean_x1 = df.loc[df['hue'] == 1]['x'].mean()
std_x1 = df.loc[df['hue'] == 1]['x'].std()
mean_y1 = df.loc[df['hue'] == 1]['y'].mean()
std_y1 = df.loc[df['hue'] == 1]['y'].std()

mean_x0 = df.loc[df['hue'] == 0]['x'].mean()
std_x0 = df.loc[df['hue'] == 0]['x'].std()
mean_y0 = df.loc[df['hue'] == 0]['y'].mean()
std_y0 = df.loc[df['hue'] == 0]['y'].std()

######################### CREATE A NORMAL DISTRIBUTION FOR BOTH 1 AND 0 ##########################################################

dist_x_1 = torch.distributions.normal.Normal(mean_x1, std_x1)
dist_y_1 = torch.distributions.normal.Normal(mean_y1, std_y1)

dist_x_0 = torch.distributions.normal.Normal(mean_x0, std_x0)
dist_y_0 = torch.distributions.normal.Normal(mean_y0, std_y0)

######################## CREATE A DIST. FOR 1 AND 0 ####################################################################################

sample_1 = torch.tensor([dist_x_1.sample(), dist_y_1.sample()]).cuda().float().view(1, 2).detach().cpu().numpy()
sample_2 = torch.tensor([dist_x_0.sample(), dist_y_0.sample()]).cuda().float().view(1, 2).detach().cpu().numpy()

diff = np.absolute(np.absolute(sample_2) - np.absolute(sample_1))
half_diff = diff / 2

new_sample = sample_1 + half_diff
new_sample = torch.tensor(new_sample).cuda().float()

new_img = ae.decode(new_sample).detach().cpu().numpy()[0]
new_img = np.moveaxis(new_img, 0, -1)
plt.imshow(new_img.reshape(28, 28), cmap='gray')
plt.show()
