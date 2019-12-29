import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

########## CREATE THE VAE MODEL #########################

class MNISTBasicVAE(nn.Module):
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
    std = torch.exp(logvar)
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

########### CREATE AN INSTANCE OF THE MODEL AND LOAD WEIGHTS ####################

vae = MNISTBasicVAE().cuda()
vae.load_state_dict(torch.load("mnist_basic_vae_weights.pth"))

############ GENERATE A NEW SAMPLE #############################

z = torch.tensor([0.65, 0.85]).view(1, 2).cuda().float()
generated_image = vae.decode(z)

############ PLOT THE NEW IMAGE ################################

generated_image = generated_image.detach().cpu().numpy()[0]
generated_image = np.moveaxis(generated_image, 0, -1)

plt.imshow(generated_image.reshape(28, 28), cmap='gray')
plt.show()
