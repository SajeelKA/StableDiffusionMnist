import sys
import os
from datetime import datetime
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import Pipeline


def inference(fileName, digit, rootDir, batch_size, embed_channels, T, conditional_scale):

  pipeline = Pipeline.Pipeline(batch_size, embed_channels, T)
  print('loading from: ', os.path.join(rootDir, 'savedModels', fileName))
  pipeline = torch.load(os.path.join(rootDir, 'savedModels', fileName), weights_only = False)  
  sampler = Pipeline.Sampling((7,7), pipeline.unet, pipeline.vae, T) #7 x 7 is input shape due to convolutional strides of 2 in encoder, will be upsampled in decoder output
  imgBatch = sampler.sample(label=digit, scale=conditional_scale, n=batch_size).squeeze(1)
  grid = torch.cat([img for img in imgBatch], dim=-1)
  plt.figure(figsize=(10,10))  
  plt.imshow(grid, cmap='gray'); plt.axis('off')
  plt.savefig(os.path.join(rootDir, 'images', 'result' + str(digit) + '.png'))
  print('figure saved to: ', os.path.join(rootDir, 'images', 'result' + str(digit) + '.png')) 


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--fileName", required = False, default = 'testStableDiffusion.pth')
  parser.add_argument("--digit", required = False, type=int, default = 3)
  parser.add_argument("--rootDir", required = False, default = '/content/drive/MyDrive/allRepos/StableDiffusionMnist/')
  parser.add_argument("--batchSize", required = False, type=int, default = 4) 
  parser.add_argument("--embedChannels", required = False, type=int, default = 16) 
  parser.add_argument("--timeSteps", required = False, type=int, default = 300) 
  parser.add_argument("--conditionalScale", required = False, type=float, default = 5.0) 
  
  args = parser.parse_args()
  inference(args.fileName, args.digit, args.rootDir, args.batchSize, args.embedChannels, args.timeSteps, args.conditionalScale)

