import sys
import os
from datetime import datetime

import Pipeline
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse


def saveModel(model, pathReq = None):

  fileName = 'model_' + str(datetime.now()).replace('-','').replace(':','').replace(' ','')[:14] + '.pth'

  if pathReq is None:
      filePath = os.path.join(os.getcwd(), fileName)
  else:
      filePath =  os.path.join(pathReq, fileName)

  torch.save(model, filePath)

  print('model saved in {}'.format(filePath))


def runTraining():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batchSize", required = False, type=int, default = 32)
  parser.add_argument("--learningRate", required = False, type=float, default = 1e-4)
  parser.add_argument("--epochs", required = False, type=int, default = 10)
  parser.add_argument("--latentEmbedChannels", required = False, type=int, default = 4) 
  parser.add_argument("--textEmbedChannels", required = False, type=int, default = 16) 
  parser.add_argument("--timeSteps", required = False, type=int, default = 300) 
  parser.add_argument("--usedProcessor", required = False, default = 'cpu')
  parser.add_argument("--savePath", required = True, default = None)

  args = parser.parse_args()
  print('\n\n' + '=' * 40 + ' Starting training with parameters: ', args, '=' * 40 + '\n\n' )
  
  
  T = args.timeSteps
  noise_channels = args.latentEmbedChannels
  embed_channels = args.textEmbedChannels
  batchSize = args.batchSize
  device = args.usedProcessor #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  lr = args.learningRate
  transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*2-1)]) # makes mnist data between 1 and -1
  ds = datasets.MNIST('.', True, transform=transform, download=True)
  loader = DataLoader(ds, batchSize, True)
  
  pipeline = Pipeline.Pipeline(noise_channels, embed_channels, T)
  opt = torch.optim.Adam(pipeline.parameters(), lr)
  
  EPOCHS = args.epochs
  
  for epoch in range(EPOCHS):
    avgLoss = 0
    avgVaeLoss = 0
    i = 0
    for imgs, labels in loader:
      imgs, labels = imgs.to(device), labels.to(device)
      total_loss, vae_loss = pipeline(imgs, labels)
      opt.zero_grad()
      total_loss.backward()
      opt.step()
      avgLoss += total_loss.item()
      avgVaeLoss += vae_loss.item()
      i += 1
  
    print(f"Epoch {epoch+1}/{EPOCHS}, avg_total_loss: {avgLoss/i:.4f}, avg_vae_loss: {avgVaeLoss/i: .4f}")
    if epoch % 5 == 0 and epoch > 0:
      saveModel(pipeline, args.savePath)
    
  saveModel(pipeline, args.savePath)

if __name__ == '__main__':
	runTraining()