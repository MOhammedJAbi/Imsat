
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

#transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
trainset = [x for x in trainset]
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
transform_test = transforms.Compose(
                                     [transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_test)
testset = [x for x in testset]
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('0','1','2','3','4','5','6','7','8','9')
tot_cl = 10

# Deep Neural Network
class MyBatchNorm(nn.Module):
    def __init__(self, num_features, bn_in, eps=1e-05, momentum=0.1, affine = False):
        super(MyBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features,eps=eps,momentum=momentum,affine=affine)
    def forward(self, x, bn_in):
        gamma = bn_in.weight
        beta = bn_in.bias
        x.data = self.bn(x) * gamma.data + beta.data
        return x

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
        self.bn1=nn.BatchNorm1d(1200)
        #self.bn1_F= MyBatchNorm(1200,self.bn1)
        self.bn1_F= nn.BatchNorm1d(1200,affine=False)
        self.bn2=nn.BatchNorm1d(1200)
        #self.bn2_F= MyBatchNorm(1200,self.bn2)
        self.bn2_F= nn.BatchNorm1d(1200,affine=False)
    
    def forward(self, x, update_batch_stats=True):
        if not update_batch_stats:
            x = x.view(-1, 28 * 28)
            x = self.fc1(x)
            x = self.bn1_F(x)*self.bn1.weight+self.bn1.bias
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2_F(x)*self.bn2.weight+self.bn2.bias
            x = F.relu(x)
            x = self.fc3(x)
            return x
        else:
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

def vat(network, x, eps_list, xi=10, Ip=1):
    # compute the regularized penality [eq. (4) & eq. (6), 1]
    with torch.no_grad():
        with torch.enable_grad():
            y = network(Variable(x))
            d = torch.randn((x.size()[0],x.size()[1],x.size()[2]*x.size()[3]))
            d = d /torch.reshape(torch.sqrt(torch.sum(torch.mul(d,d), dim=2)),(x.size()[0],x.size()[1],1))
            d = torch.reshape(d,(x.size()[0],x.size()[1],x.size()[2],x.size()[3]))
            for ip in range(Ip):
                d_var = Variable(d)
                if use_cuda:
                    d_var = d_var.to(device)
                d_var.requires_grad_(True)
                y_p = network(x + xi * d_var)
                kl_loss = kl(F.softmax(y,dim=1), F.softmax(y_p,dim=1))
                #kl_loss.backward(retain_graph=True)
                kl_loss.backward()
                d = d_var.grad
                d = torch.reshape(d,(x.size()[0],x.size()[1],x.size()[2]*x.size()[3]))
                norm = torch.sqrt(torch.sum(torch.mul(d,d), dim=2))
                norm[norm==0] = 1
                d = d /torch.reshape(norm,(x.size()[0],x.size()[1],1))
                d = torch.reshape(d,(x.size()[0],x.size()[1],x.size()[2],x.size()[3]))
    delta = d 
    y = network(Variable(x))
    eps = args.prop_eps * eps_list
    y2 = network(x + torch.reshape(eps,(x.size()[0],1,1,1)) * delta)
    return kl(F.softmax(y,dim=1), F.softmax(y2,dim=1))

def enc_aux_noubs(x):
    # not updating gamma and beta in batchs
    return net(x, update_batch_stats=False)

def loss_unlabeled(x, eps_list):
    # to use enc_aux_noubs
    L = vat(enc_aux_noubs, x, eps_list)
    return L

def upload_nearest_dist(batch_size,loader_number,dataset):
    # Import the range of local perturbation for VAT
    nearest_dist = np.loadtxt(dataset + '/' + 'loader_number'+ str(loader_number+1) + '_' 'batch_size'+ str(batch_size) + '_10th_neighbor.txt').astype(np.float32)
    return nearest_dist

optimizer = optim.Adam(net.parameters(), lr=lr)


def bestMap(L1, L2):
    # compute the accuracy using Hungarian algorithm
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
    running_loss = 0.0
    sum_aver_entropy = 0
    sum_entropy_aver = 0
    vatt = 0
    #   start_time = time.clock()
    for i, data in enumerate(trainloader, 0):
        
        # get the inputs
        inputs, labels = data
        nearest_dist = torch.from_numpy(upload_nearest_dist(batch_size,i,args.dataset))
        if use_cuda:
            inputs, labels, nearest_dist= inputs.to(device), labels.to(device), nearest_dist.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward
        aver_entropy, entropy_aver = Compute_entropy(net, Variable(inputs))
        r_mutual_i = aver_entropy - args.mu * entropy_aver
        loss_ul = loss_unlabeled(inputs, nearest_dist)
        loss = loss_ul + args.lam * r_mutual_i
        
        # backward + optimize
        loss.backward()
        optimizer.step()
        
        loss_ul.detach_()
        
        # loss accumulation
        sum_aver_entropy += aver_entropy
        sum_entropy_aver += entropy_aver
        #vatt += loss_ul.item()
        running_loss += loss.item()
    print("running_loss: ",running_loss/(i+1))

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


