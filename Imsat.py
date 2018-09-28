
"""
Pytorch code to reproduce the results of the clustring algorithm IMSAT [1]. The adoped data set is MNIST. The implementation in [1] is based on Chainer.

[1] Weihua Hu, Takeru Miyato, Seiya Tokui, Eiichi Matsumoto and Masashi Sugiyama. Learning Discrete Representations via Information Maximizing Self-Augmented Training. In ICML, 2017. Available at http://arxiv.org/abs/1702.08720
"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.utils import linear_assignment_
from sklearn.metrics import accuracy_score
from munkres import Munkres, print_matrix
import itertools

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--batch_size', '-b', default=250, type=int, help='size of the batch during training')
parser.add_argument('--lam', type=float, help='trade-off parameter for mutual information and smooth regularization',default=0.1)
parser.add_argument('--mu', type=float, help='trade-off parameter for entropy minimization and entropy maximization',default=4)
parser.add_argument('--prop_eps', type=float, help='epsilon', default=0.25)
parser.add_argument('--hidden_list', type=str, help='hidden size list', default='1200-1200')
parser.add_argument('--n_epoch', type=int, help='number of epoches when maximizing', default=50)
parser.add_argument('--dataset', type=str, help='which dataset to use', default='mnist')
args = parser.parse_args()

batch_size = args.batch_size
hidden_list = args.hidden_list
lr = args.lr
n_epoch = args.n_epoch



# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Data
print('==> Preparing data..')
class MyDataset(torch.utils.data.Dataset):
    # new dataset class that allows to get the sample indices of mini-batch
    def __init__(self,root,download, train, transform):
        self.MNIST = torchvision.datasets.MNIST(root=root,
                                        download=download,
                                        train=train,
                                        transform=transform)
    def __getitem__(self, index):
        data, target = self.MNIST[index]
        return data, target, index
    
    def __len__(self):
        return len(self.MNIST)

transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
trainset = MyDataset(root='./data', train=True, download=True, transform=transform_train)
testset = MyDataset(root='./data', train=False, download=False, transform=transform_train)
trainset = trainset + testset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
tot_cl = 10

# Deep Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1200)
        torch.nn.init.normal_(self.fc1.weight,std=0.1*math.sqrt(2/(28*28)))
        self.fc1.bias.data.fill_(0)
        self.fc2 = nn.Linear(1200, 1200)
        torch.nn.init.normal_(self.fc2.weight,std=0.1*math.sqrt(2/1200))
        self.fc2.bias.data.fill_(0)
        self.fc3 = nn.Linear(1200, 10)
        torch.nn.init.normal_(self.fc3.weight,std=0.0001*math.sqrt(2/1200))
        self.fc3.bias.data.fill_(0)
        self.bn1=nn.BatchNorm1d(1200, eps=2e-5)
        self.bn1_F= nn.BatchNorm1d(1200, eps=2e-5, affine=False)
        self.bn2=nn.BatchNorm1d(1200, eps=2e-5)
        self.bn2_F= nn.BatchNorm1d(1200, eps=2e-5, affine=False)
    
    def forward(self, x, update_batch_stats=True):
        if not update_batch_stats:
            x = self.fc1(x)
            x = self.bn1_F(x)*self.bn1.weight+self.bn1.bias
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2_F(x)*self.bn2.weight+self.bn2.bias
            x = F.relu(x)
            x = self.fc3(x)
            return x
        else:
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.fc3(x)
            return x

net = Net()
if use_cuda:
    net.to(device)

# Loss function and optimizer
def entropy(p):
    # compute entropy
    if (len(p.size())) == 2:
        return - torch.sum(p * torch.log(p + 1e-8)) / float(len(p))
    elif (len(p.size())) == 1:
        return - torch.sum(p * torch.log(p + 1e-8))
    else:
        raise NotImplementedError

def Compute_entropy(net, x):
    # compute the entropy and the conditional entropy
    p = F.softmax(net(x),dim=1)
    p_ave = torch.sum(p, dim=0) / len(x)
    return entropy(p), entropy(p_ave)

def kl(p, q):
    # compute KL divergence between p and q
    return torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8))) / float(len(p))

def distance(y0, y1):
    # compute KL divergence between the outputs of the newtrok
    return kl(F.softmax(y0,dim=1), F.softmax(y1,dim=1))

def vat(network, distance, x, eps_list, xi=10, Ip=1):
    # compute the regularized penality [eq. (4) & eq. (6), 1]
    
    with torch.no_grad():
        y = network(Variable(x))
    d = torch.randn((x.size()[0],x.size()[1]))
    d = F.normalize(d, p=2, dim=1)
    for ip in range(Ip):
        d_var = Variable(d)
        if use_cuda:
            d_var = d_var.to(device)
        d_var.requires_grad_(True)
        y_p = network(x + xi * d_var)
        kl_loss = distance(y,y_p)
        kl_loss.backward(retain_graph=True)
        d = d_var.grad
        d = F.normalize(d, p=2, dim=1)
    d_var = d
    if use_cuda:
        d_var = d_var.to(device)
    eps = args.prop_eps * eps_list
    eps = eps.view(-1,1)
    y_2 = network(x + eps*d_var)
    return distance(y,y_2)

def enc_aux_noubs(x):
    # not updating gamma and beta in batchs
    return net(x, update_batch_stats=False)

def loss_unlabeled(x, eps_list):
    # to use enc_aux_noubs
    L = vat(enc_aux_noubs, distance, x, eps_list)
    return L

def upload_nearest_dist(dataset):
    # Import the range of local perturbation for VAT
    nearest_dist = np.loadtxt(dataset + '/' + '10th_neighbor.txt').astype(np.float32)
    return nearest_dist

optimizer = optim.Adam(net.parameters(), lr=lr)

def compute_accuracy(y_pred, y_t):
    # compute the accuracy using Hungarian algorithm
    m = Munkres()
    mat = np.zeros((tot_cl, tot_cl))
    for i in range(tot_cl):
        for j in range(tot_cl):
            mat[i][j] = np.sum(np.logical_and(y_pred == i, y_t == j))
    indexes = m.compute(-mat)

    corresp = []
    for i in range(tot_cl):
        corresp.append(indexes[i][1])

    pred_corresp = [corresp[int(predicted)] for predicted in y_pred]
    acc = np.sum(pred_corresp == y_t) / float(len(y_t))
    return acc


# Training
print('==> Start training..')
nearest_dist = torch.from_numpy(upload_nearest_dist(args.dataset))
if use_cuda:
    nearest_dist = nearest_dist.to(device)

for epoch in range(n_epoch):
    net.train()
    running_loss = 0.0
    best_acc = 0
    #   start_time = time.clock()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels, ind = data
        inputs = inputs.view(-1, 28 * 28)
        if use_cuda:
            inputs, labels, nearest_dist, ind = inputs.to(device), labels.to(device), nearest_dist.to(device), ind.to(device)
        
        # forward
        aver_entropy, entropy_aver = Compute_entropy(net, Variable(inputs))
        r_mutual_i = aver_entropy - args.mu * entropy_aver
        loss_ul = loss_unlabeled(Variable(inputs), nearest_dist[ind])
        
        loss = loss_ul + args.lam * r_mutual_i
      
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # backward + optimize
        loss.backward()
        optimizer.step()
        
        # loss accumulation
        running_loss += loss.item()

    # statistics
    net.eval()
    p_pred = np.zeros((len(trainset),10))
    y_pred = np.zeros(len(trainset))
    y_t = np.zeros(len(trainset))
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels, ind = data
            inputs = inputs.view(-1, 28 * 28)
            if use_cuda:
                inputs, labels = inputs.to(device), labels.to(device)
            outputs=F.softmax(net(inputs),dim=1)
            y_pred[i*batch_size:(i+1)*batch_size]=torch.argmax(outputs,dim=1).cpu().numpy()
            p_pred[i*batch_size:(i+1)*batch_size,:]=outputs.detach().cpu().numpy()
            y_t[i*batch_size:(i+1)*batch_size]=labels.cpu().numpy()
    acc = compute_accuracy(y_pred, y_t)
    print("epoch: ", epoch+1, "\t total lost = {:.4f} " .format(running_loss/(i+1)), "\t MI = {:.4f}" .format(normalized_mutual_info_score(y_t, y_pred)), "\t acc = {:.4f} " .format(acc))

    # save the "best" parameters
    if acc > best_acc:
        best_acc = acc
    # show results
print("Best accuracy = {:.4f}" .format(best_acc))
print('==> Finished Training..')

