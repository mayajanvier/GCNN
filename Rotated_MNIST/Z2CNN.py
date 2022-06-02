from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

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

# Baseline Z2CNN 
class Z2CNN(nn.Module): 
  def __init__( self, input_size=28*28, input_channels=1, n_feature=20, output_size=10):
    """
    :param input_size: number of pixels in the image
    :param input_channels: number of color channels in the image
    :param n_feature: size of the hidden dimensions to use
    :param output_size: expected size of the output
    """
    ksize = 3
    ksize_f = 4

    super().__init__()
    self.n_feature = n_feature
    self.conv1 = nn.Conv2d(
        in_channels=input_channels, out_channels=n_feature, kernel_size=ksize,padding=1
    )
    self.bn1 = nn.BatchNorm2d(n_feature)
    self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=ksize,padding=1) #need to add padding to keep the same dimensions of the image 
    self.bn2 = nn.BatchNorm2d(n_feature)
    self.conv3 = nn.Conv2d(n_feature, n_feature, kernel_size=ksize,padding=1)
    self.bn3 = nn.BatchNorm2d(n_feature)
    self.conv4 = nn.Conv2d(n_feature, n_feature, kernel_size=ksize,padding=1)
    self.bn4 = nn.BatchNorm2d(n_feature)
    self.conv5 = nn.Conv2d(n_feature, n_feature, kernel_size=ksize,padding=1)
    self.bn5 = nn.BatchNorm2d(n_feature)
    self.conv6 = nn.Conv2d(n_feature, n_feature, kernel_size=ksize,padding=1)
    self.bn6 = nn.BatchNorm2d(n_feature)
    self.conv7 = nn.Conv2d(n_feature, output_size, kernel_size=ksize_f,padding=1)
    #self.bn7 = nn.BatchNorm2d(n_feature)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout2d = nn.Dropout2d(p=0.3)  
    self.fc1 = nn.Linear(1690, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, 
              x: torch.Tensor
      ) -> torch.Tensor:
      """
      :param x: batch of images with size [batch, 1, w, h]
      :returns: predictions with size [batch, output_size]
      """
      x = F.relu(self.bn1(self.dropout2d(self.conv1(x))))
      x = F.relu(self.bn2(self.dropout2d(self.conv2(x))))
      x = F.max_pool2d(x, kernel_size=2)
      x = F.relu(self.bn3(self.dropout2d(self.conv3(x))))
      x = F.relu(self.bn4(self.dropout2d(self.conv4(x))))
      x = F.relu(self.bn5(self.dropout2d(self.conv5(x))))
      x = F.relu(self.bn6(self.dropout2d(self.conv6(x))))
      x = self.conv7(x)
      x = x.view(x.shape[0],-1) 
      x = F.relu(self.fc1(x))
      x = F.dropout(x, training=self.training)
      x = self.fc2(x)
      return F.log_softmax(x)*
    
    
#Model definition
print('Z2CNN')
model = Z2CNN()
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
