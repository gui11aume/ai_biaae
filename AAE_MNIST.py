#!/usr/bin/env python
# -*- coding:utf-8 -*-


import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

import sys

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.data as Data

import torchvision

from torch.distributions.beta import Beta
from torch.distributions.constraints import positive
from torch.distributions.normal import Normal


def InverseLinear(x):
   # Inverse-Linear activation function
   return 1.0 / (1.0-x+torch.abs(x)) + x+torch.abs(x)


class Encoder(nn.Module):

   def __init__(self, lyrsz):
      super(Encoder, self).__init__()
      self.lyrsz = lyrsz
      self.lyrs = nn.ModuleList()
      for isz, osz in zip(lyrsz[0:], lyrsz[1:]):
         self.lyrs.append(nn.Linear(isz, osz))

   def forward(self, x):
      for lyr in self.lyrs[:-1]:
         x = torch.relu(lyr(x))
      return self.lyrs[-1](x)


class Decoder(nn.Module):

   def __init__(self, lyrsz):
      super(Decoder, self).__init__()
      self.lyrsz = lyrsz
      self.lyrs = nn.ModuleList()
      for isz, osz in zip(lyrsz[0:-1], lyrsz[1:-1]):
         self.lyrs.append(nn.Linear(isz, osz))
      self.alyr = nn.Linear(lyrsz[-2], lyrsz[-1])
      self.blyr = nn.Linear(lyrsz[-2], lyrsz[-1])

   def forward(self, x):
      for lyr in self.lyrs:
         x = torch.relu(lyr(x))
      a = torch.clamp(InverseLinear(self.alyr(x)/2), min=.01, max=100)
      b = torch.clamp(InverseLinear(self.blyr(x)/2), min=.01, max=100)
      return (a,b)


class Checker(nn.Module):

   def __init__(self, zlyrsz):
      super(Checker, self).__init__()
      self.lyrC1 = nn.Linear(zlyrsz, 128)
      self.lyrC2 = nn.Linear(128, 128)
      self.lyrC3 = nn.Linear(128, 1)

   def forward(self, x):
      x = torch.relu(self.lyrC1(x))
      x = torch.relu(self.lyrC2(x))
      return torch.sigmoid(self.lyrC3(x))


class AdversarialAutoEncoder(nn.Module):

   def __init__(self, zlyrsz=20):
      super(AdversarialAutoEncoder, self).__init__()
      # Store arguments
      self.zlyrsz = zlyrsz
      self.checker = Checker(zlyrsz)
      self.encoder = Encoder([784, 1200, 600, 300, zlyrsz])
      self.decoder = Decoder([zlyrsz, 300, 600, 1200, 784])

   def forward(self, x):
      z = self.encoder(x)
      (a,b) = self.decoder(z)
      return a, b, z

   def model(self, obs, idx):
      sz = torch.Size((obs.shape[0], self.zlyrsz))
      mu = obs.new_zeros(sz)
      sd = obs.new_ones(sz)
      with pyro.plate('batch', obs.shape[0], device=obs.device):
         z = pyro.sample('z', dist.Normal(mu,sd).to_event(1))
         (a,b) = self.decoder(z)
         ax = a[:,idx]
         bx = b[:,idx]
         pyro.sample('x', dist.Beta(ax, bx).to_event(1), obs=obs)

   def guide(self, obs, idx):
      sz = torch.Size((obs.shape[0], self.zlyrsz))
      mu = pyro.param("mu", obs.new_zeros(sz))
      sd = pyro.param("sd", obs.new_ones(sz), constraint=positive)
      with pyro.plate('batch', obs.shape[0], device=obs.device):
         pyro.sample('z', dist.Normal(mu,sd).to_event(1))

   def optimize(self, train_data, epochs, batch_size=1024):
      lossf=func.binary_cross_entropy

      encoder_optimizer = \
            torch.optim.Adam(self.encoder.parameters(), lr=0.001)
      decoder_optimizer = \
            torch.optim.Adam(self.decoder.parameters(), lr=0.001)
      checker_optimizer = \
            torch.optim.Adam(self.checker.parameters(), lr=0.001)

      encoder_sched = torch.optim.lr_scheduler.MultiStepLR(
            encoder_optimizer, [150,250])
      decoder_sched = torch.optim.lr_scheduler.MultiStepLR(
            decoder_optimizer, [150,250])
      checker_sched = torch.optim.lr_scheduler.MultiStepLR(
            checker_optimizer, [150,250])

      batches = Data.DataLoader(dataset=train_data,
            batch_size=batch_size, shuffle=True)

      for e in range(epochs):
         b_rcst = 0.0
         b_fool = 0.0
         b_disc = 0.0
         for batch_no, data in enumerate(batches):
            target_real = data.new_ones(len(data), 1)
            target_fake = data.new_zeros(len(data) ,1)
            target_disc = torch.cat((target_fake, target_real), 0)
            # Phase I: REGULARIZATION.
            # An important point is to tune the cost of the
            # regularization versus the reconstruction. This is
            # done by diluting the loss of the checker and the
            # fooler.
            with torch.no_grad():
               z_fake = self.encoder(data)
               z_real = torch.randn(len(data), self.zlyrsz,
                     device=data.device)
            disc = self.checker(torch.cat((z_fake, z_real), 0))
            disc_loss = lossf(disc, target_disc) / 10 # <---
            b_disc += float(disc_loss)
            # Reset gradients.
            checker_optimizer.zero_grad()
            disc_loss.backward()
            # Update discriminator parameters
            checker_optimizer.step()
            # Phase II. RECONSTRUCTION.
            a, b, z = self(data)
            fool = self.checker(z)
            fool_loss = lossf(fool, target_real) / 10 # <---
            l = Beta(a,b).log_prob(torch.clamp(data, min=.001, max=.999))
            rcst_loss = -torch.mean(l)
            loss = rcst_loss + fool_loss
            b_rcst += float(rcst_loss)
            b_fool += float(fool_loss)
            # Reset gradients.
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            checker_optimizer.zero_grad()
            loss.backward()
            # Update discriminator parameters
            encoder_optimizer.step()
            decoder_optimizer.step()
         encoder_sched.step()
         decoder_sched.step()
         checker_sched.step()

         # Report.
         print e, b_rcst
         print e, b_fool
         print e, b_disc
         print '---'


train = torchvision.datasets.MNIST('./', train=True, download=True,
      transform=torchvision.transforms.ToTensor())
train_data = train.data.view(-1,28*28).to(device='cuda',
      dtype=torch.float32)/255.0

AAE = AdversarialAutoEncoder().cuda()
AAE.optimize(train_data, epochs=300)

# Save / load the trained network as required.
torch.save(AAE.state_dict(), 'AAE_MNIST_state_dict.tch')
AAE.load_state_dict(torch.load('AAE_MNIST_state_dict.tch'))

test = torchvision.datasets.MNIST('./', train=False, download=True,
      transform=torchvision.transforms.ToTensor())
test_data  = test.data.view(-1,28*28).to(device='cuda',
      dtype=torch.float32)/255.0

# Number reconstruction
samples = 30
iters = 28

test_samples = test_data[:samples,:].detach().clone()
obs = torch.clamp(test_samples[:,:392], min=.001, max=.999)

orig = test_samples.to('cpu').view(-1,28)
half = test_samples.clone()
half[:,392:] = 0.0
half = half.view(-1,28)

image = np.zeros((28*samples, 28*(iters+2)))
image[:,:28] = orig
image[:,28:56] = half.to('cpu')

idx = np.arange(392)

optim = pyro.optim.Adam({"lr": 0.01})
optim = torch.optim.Adam
sched = pyro.optim.MultiStepLR({
            'optimizer': optim,
            'milestones': [10000, 20000],
            'optim_args': {'lr': 0.01}
        })
svi = pyro.infer.SVI(AAE.model, AAE.guide,
         sched, loss=pyro.infer.Trace_ELBO())

# Iterate
for it in np.arange(iters)+2:
   for step in range(1000):
      svi.step(obs, idx)
   mu = pyro.param("mu").view(-1,20)
   sd = pyro.param("sd").view(-1,20)
   with torch.no_grad():
      smplz = Normal(mu,sd).sample()
      (a,b) = AAE.decoder(smplz)
      x = sum([Beta(a,b).sample() / 5 for _ in range(5)])
   print 'frame', it-1
   x[:,:392] = test_samples[:,:392]
   image[:,(it*28):((it+1)*28)] = x.to('cpu').view(-1,28).numpy()

plt.figure(figsize=(10,10))
plt.imshow(1-image, cmap='gray')
plt.savefig('AAE_reconstruct.png')
