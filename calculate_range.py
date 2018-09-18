"""
    Compute the euclidian distance to the 10-th neighbor used to estimate epsilon; see [eq.(4) & eq. (17), 1]
    
    [1] Weihua Hu, Takeru Miyato, Seiya Tokui, Eiichi Matsumoto and Masashi Sugiyama. Learning Discrete Representations via Information Maximizing Self-Augmented Training. In ICML, 2017. Available at http://arxiv.org/abs/1702.08720
"""

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
parser.add_argument('--batch_size', '-b', default=250, type=int, help='size of the batch during training')
parser.add_argument('--dataset', type=str, default = 'mnist')

args = parser.parse_args()

batch_size = args.batch_size

# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
trainset = [x for x in trainset]

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

transform_test = transforms.Compose(
                                    [transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testset = [x for x in testset]
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

loader = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=False, num_workers=2)

classes = ('0','1','2','3','4','5','6','7','8','9')

for j, data_t in enumerate(loader, 0):
    # get all inputs
    inputs_t, labels_t = data_t
    if use_cuda:
        inputs_t, labels_t = inputs_t.to(device), labels_t.to(device)

num_data = [10]

print(len(num_data))

for j, data in enumerate(trainloader, 0):
    dist_list = [[] for i in range(len(num_data))]
    if j >=0:
        print(j)
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        for i in range(len(inputs)):
            aa = torch.mul(inputs_t - inputs[i],inputs_t - inputs[i])
            dist = torch.sqrt(torch.sum(aa,dim=(2,3)))
            dist_m = dist[:,0]
            dist_m[i] = 1000
            sorted_dist, indices = torch.sort(dist_m)
            for jj in range(len(num_data)):
                dist_list[jj].append(sorted_dist[num_data[jj]])
        for ii in range(len(num_data)):
                np.savetxt(args.dataset + '/' + 'loader_number'+ str(j+1) + '_' 'batch_size'+ str(batch_size) + '_' + str(num_data[ii]) + 'th_neighbor.txt', np.array(dist_list[ii]))