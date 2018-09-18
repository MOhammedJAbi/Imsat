
"""
Pytorch code to reproduce the results of the clustring algorithm Deep_RIM [1]. The adoped data set is MNIST. The implementation in [1] is based on Chainer.

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

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--batch_size', '-b', default=250, type=int, help='size of the batch during training')
parser.add_argument('--lam', type=float, help='trade-off parameter for mutual information and smooth regularization',default=0.1)
parser.add_argument('--mu', type=float, help='trade-off parameter for entropy minimization and entropy maximization',default=1)
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

#transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
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

classes = ('0','1','2','3','4','5','6','7','8','9')
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
        self.bn1=nn.BatchNorm1d(1200, affine=True)
        self.bn2=nn.BatchNorm1d(1200, affine=True)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
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
        return - torch.sum(p * torch.log(p + 1e-18)) / float(len(p))
    elif (len(p.size())) == 1:
        return - torch.sum(p * torch.log(p + 1e-18))
    else:
        raise NotImplementedError

def Compute_entropy(net, x):
    # compute the entropy and the conditional entropy
    p = F.softmax(net(x),dim=1)
    p_ave = torch.sum(p, dim=0) / len(x)
    return entropy(p), entropy(p_ave)

def kl(p, q):
    # compute KL divergence between p and q
    return torch.sum(p * torch.log((p + 1e-18) / (q + 1e-18))) / float(len(p))

optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.005)


def bestMap(L1, L2):
    # compute the accuracy
    if L1.__len__() != L2.__len__():
        print('size(L1) must == size(L2)')
    
    Label1 = np.unique(L1)
    nClass1 = Label1.__len__()
    Label2 = np.unique(L2)
    nClass2 = Label2.__len__()
    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i][j] = np.nonzero((L1 == Label1[i]) * (L2 == Label2[j]))[0].__len__()

    c = linear_assignment_.linear_assignment(-G.T)[:, 1]
    newL2 = np.zeros(L2.__len__())
    for i in range(nClass2):
        for j in np.nonzero(L2 == Label2[i])[0]:
            if len(Label1) > c[i]:
                newL2[j] = Label1[c[i]]

    return accuracy_score(L1, newL2)

# Training
print('==> Start training..')
#net.train()
for epoch in range(n_epoch):
    net.train()
    print("epoch: ",epoch)
    running_loss = 0.0
    sum_aver_entropy = 0
    sum_entropy_aver = 0
    vatt = 0
    #   start_time = time.clock()
    for i, data in enumerate(trainloader, 0):
        
        # get the inputs
        inputs, labels = data
        #nearest_dist = torch.from_numpy(upload_nearest_dist(batch_size,i,args.dataset))
        if use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        #inputs, labels, nearest_dist= inputs.to(device), labels.to(device), nearest_dist.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward
        aver_entropy, entropy_aver = Compute_entropy(net, Variable(inputs))
        loss = aver_entropy - entropy_aver
        
        # backward + optimize
        loss.backward()
        optimizer.step()
        
        # loss accumulation
        sum_aver_entropy += aver_entropy
        sum_entropy_aver += entropy_aver
        #vatt += loss_ul.item()
        running_loss += loss.item()

    # statistics
    net.eval()
    p_pred = np.zeros((len(trainset),10))
    y_pred = np.zeros(len(trainset))
    y = np.zeros(len(trainset))
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        
        outputs=F.softmax(net(inputs),dim=1)
        y_pred[i*batch_size:(i+1)*batch_size]=torch.argmax(outputs,dim=1).cpu().numpy()
        p_pred[i*batch_size:(i+1)*batch_size,:]=outputs.detach().cpu().numpy()
        y[i*batch_size:(i+1)*batch_size]=labels.cpu().numpy()

    print("epoch: ", epoch+1, "\t total lost = {:.4f} " .format(running_loss/(i+1)), "\t MI {:.4f}" .format(normalized_mutual_info_score(y, y_pred)), "\t acc = {:.4f} " .format(bestMap(y, y_pred)))
#    print(time.clock() - start_time, "seconds")
print('==> Finished Training..')

