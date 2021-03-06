from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# P4 layers from GrouPy 
from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Rotated datasets for training and test 
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       transforms.RandomRotation(90)
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader_rot = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       transforms.RandomRotation(90)
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# Non rotated test dataset for comparison
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))#,
                       #transforms.RandomRotation(90)
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# Different versions of P4CNN 
class P4CNN_drop(nn.Module):
    """ P4CNN without the last max pooling layer, with dropout """
    def __init__(self):
        super(P4CNN_drop, self).__init__()
        self.conv1 = P4ConvZ2(1, 10, kernel_size=3)
        self.conv2 = P4ConvP4(10, 10, kernel_size=3)
        self.conv3 = P4ConvP4(10, 10, kernel_size=3)
        self.conv4 = P4ConvP4(10, 10, kernel_size=3)
        self.conv5 = P4ConvP4(10, 10, kernel_size=3)
        self.conv6 = P4ConvP4(10, 10, kernel_size=3)
        self.conv7 = P4ConvP4(10, 10, kernel_size=4)
        self.fc1 = nn.Linear(40, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class P4CNN_no_drop(nn.Module):
    """ P4CNN without the last max pooling layer, without dropout """
    def __init__(self):
        super(P4CNN_no_drop, self).__init__()
        self.conv1 = P4ConvZ2(1, 10, kernel_size=3)
        self.conv2 = P4ConvP4(10, 10, kernel_size=3)
        self.conv3 = P4ConvP4(10, 10, kernel_size=3)
        self.conv4 = P4ConvP4(10, 10, kernel_size=3)
        self.conv5 = P4ConvP4(10, 10, kernel_size=3)
        self.conv6 = P4ConvP4(10, 10, kernel_size=3)
        self.conv7 = P4ConvP4(10, 10, kernel_size=4)
        self.fc1 = nn.Linear(40, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class P4CNN_max(nn.Module): 
    """ P4CNN with the last max pooling layer, without dropout """
    def __init__(self):
        super(P4CNN_max, self).__init__()
        self.conv1 = P4ConvZ2(1, 10, kernel_size=3)
        self.conv2 = P4ConvP4(10, 10, kernel_size=3)
        self.conv3 = P4ConvP4(10, 10, kernel_size=3)
        self.conv4 = P4ConvP4(10, 10, kernel_size=3)
        self.conv5 = P4ConvP4(10, 10, kernel_size=3)
        self.conv6 = P4ConvP4(10, 10, kernel_size=3)
        self.conv7 = P4ConvP4(10, 10, kernel_size=3)
        self.fc1 = nn.Linear(40, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
#Model definition
print('Best version of P4CNN: without last max pooling layer, without dropout')
model = P4CNN_no_drop()
if args.cuda:
    model.cuda()
    
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) #if you want to try Adam as optimizer 

# Training and testing routines 
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def test():
    """ Test the performances on both rotated and non rotated test sets"""
    print('test non rotated')
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


    print('test rotated')
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader_rot:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    
# Training and testing
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
    
# Other versions    
print('P4CNN without last max pooling layer, with dropout')
model = P4CNN_drop()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
if args.cuda:
    model.cuda()
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
    
 
print('P4CNN with last max pooling layer, without dropout')
model = P4CNN_max()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
if args.cuda:
    model.cuda()
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
        
