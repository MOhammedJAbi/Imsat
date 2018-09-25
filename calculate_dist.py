import os
import torch
import torchvision
import torchvision.transforms as transforms
import time
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import cuda

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'mnist')
args = parser.parse_args()

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_train)
trainset = trainset + testset
loader = torch.utils.data.DataLoader(trainset, batch_size=70000, shuffle=False, num_workers=2)
# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_data = [10]
with torch.no_grad():
    for j, data_t in enumerate(loader, 0):
        dist_list = [[] for i in range(len(num_data))]
    # get all inputs
        inputs_t, labels_t = data_t
        if use_cuda:
            inputs_t, labels_t = inputs_t.to(device), labels_t.to(device)
        for i in range(len(inputs_t)):
            if i%1000 == 0:
                print(i)
            aa = torch.mul(inputs_t - inputs_t[i],inputs_t - inputs_t[i])
            dist = torch.sqrt(torch.sum(aa,dim=(2,3)))
            dist_m = dist[:,0]
            dist_m[i] = 1000
            sorted_dist = np.sort(dist_m.cpu().numpy())
            for jj in range(len(num_data)):
                dist_list[jj].append(sorted_dist[num_data[jj]])
    for ii in range(len(num_data)):
        np.savetxt(args.dataset + '/' + str(num_data[ii]) + 'th_neighbor.txt', np.array(dist_list[ii]))