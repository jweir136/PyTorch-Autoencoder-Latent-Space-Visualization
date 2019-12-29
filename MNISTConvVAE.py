import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import math
import numpy as np
import matplotlib.pyplot as plt

################### SPECIFY THE DIRECTORIES AND TRANSFORMATIONS ###############################

TRAINING_DIR = "/storage/mnist-jpg/trainingSet/trainingSet"
TEST_DIR = "/storage/mnist-jpg/testSet"

trans = transforms.Compose([
  transforms.Grayscale(),
  transforms.Resize(28),
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.5,), std=(0.5,))
])

################### CREATE THE DATASET OBJECT #################################################

trainfolder = datasets.ImageFolder(root=TRAINING_DIR, transform=trans)
testfolder = datasets.ImageFolder(root=TEST_DIR, transform=trans)

trainloader = data.DataLoader(trainfolder, batch_size=128, shuffle=True, num_workers=12)
testloader = data.DataLoader(testfolder, batch_size=128, shuffle=True, num_workers=12)

################### CREATE THE AUTOENCODER ####################################################

class Flatten(nn.Module):
  def forward(self, x):
    batch_size = x.size()[0]
    return x.view(batch_size, -1)

class UnFlatten(nn.Module):
  def forward(self, x):
    batch_size = x.size()[0]
    return x.view(batch_size, 16, 20, 20)

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

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparam_(mu, logvar)
    x = self.decode(z)
    return x, mu, logvar

################ CREATE THE LOSS FUNCTION, OPTIMIZER, AND THE MODEL ##########################

def loss_function(x_pred, x, mu, logvar):
  mse = fn.mse_loss(x_pred, x)
  kl = -5e-4 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
  return mse + kl

ae = MNISTAE().cuda()
adam = optim.SGD(ae.parameters(), lr=1e-3, momentum=0.9)

################ TRAIN THE MODEL #############################################################

for epoch in range(50):
  for x, _ in trainloader:
    x = x.cuda().float()
    
    adam.zero_grad()

    x_pred, mu, logvar = ae.forward(x)

    train_loss = loss_function(x_pred, x, mu, logvar)
    train_loss.backward()
    adam.step()

  for x, _ in testloader:
    x = x.cuda().float()

    x_pred, mu, logvar = ae.forward(x)

    test_loss = loss_function(x_pred, x, mu, logvar)
 
  pred_img = np.moveaxis(x_pred.detach().cpu().numpy()[0], 0, -1).reshape(28, 28)
  img = np.moveaxis(x.detach().cpu().numpy()[0], 0, -1).reshape(28, 28) 

  fig = plt.figure()

  ax1 = fig.add_subplot(2, 1, 1)
  ax1.imshow(pred_img, cmap='gray')
  ax2 = fig.add_subplot(2, 1, 2)
  ax2.imshow(img, cmap='gray')
  fig.savefig("/artifacts/mnist_conv_vae_images/generated_image_epoch_{}.png".format(epoch+1))

  print("\n")
  print("[{}] Train Loss={} Test Loss={}".format(epoch+1, train_loss.detach().cpu().numpy(), test_loss.detach().cpu().numpy()))
  print("\n")

################# SAVE THE IMAGE ##############################################################

torch.save(ae.state_dict(), "/artifacts/mnist_conv_vae_weights.pth")
