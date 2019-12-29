import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

################### SPECIFY THE DIRECTORIES AND TRANSFORMATIONS ###############################

TRAINING_DIR = "../../datasets/mnist-jpg/trainingSet/trainingSet"
TEST_DIR = "../../datasets/mnist-jpg/testSet"

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

class MNISTAE(nn.Module):
  def __init__(self):
    super().__init__()

    self.encoder = nn.Sequential(
      nn.Conv2d(1, 8, kernel_size=5),
      nn.ReLU(True),
      nn.Conv2d(8, 16, kernel_size=5),
      nn.ReLU(True)
    )
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(16, 8, kernel_size=5),
      nn.ReLU(True),
      nn.ConvTranspose2d(8, 1, kernel_size=5),
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

################ CREATE THE LOSS FUNCTION, OPTIMIZER, AND THE MODEL ##########################

ae = MNISTAE().cuda()
loss_function = nn.MSELoss()
adam = optim.SGD(ae.parameters(), lr=1e-3, momentum=0.9)

################ TRAIN THE MODEL #############################################################

for epoch in range(50):
  for x, _ in tqdm(trainloader):
    x = x.cuda().float()
    
    adam.zero_grad()

    x_pred = ae.forward(x)

    train_loss = loss_function(x_pred, x)
    train_loss.backward()
    adam.step()

  for x, _ in tqdm(testloader):
    x = x.cuda().float()

    x_pred = ae.forward(x)

    test_loss = loss_function(x_pred, x)
 
  pred_img = np.moveaxis(x_pred.detach().cpu().numpy()[0], 0, -1).reshape(28, 28)
  img = np.moveaxis(x.detach().cpu().numpy()[0], 0, -1).reshape(28, 28) 

  fig = plt.figure()

  ax1 = fig.add_subplot(2, 1, 1)
  ax1.imshow(pred_img, cmap='gray')
  ax2 = fig.add_subplot(2, 1, 2)
  ax2.imshow(img, cmap='gray')
  fig.savefig("mnist_conv_autoencoder_images/generated_image_epoch_{}.png".format(epoch+1))

  print("\n")
  print("[{}] Train Loss={} Test Loss={}".format(epoch+1, train_loss.detach().cpu().numpy(), test_loss.detach().cpu().numpy()))
  print("\n")

################# SAVE THE IMAGE ##############################################################

torch.save(ae.state_dict(), "mnist_conv_autoencoder_weights.pth")
