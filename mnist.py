from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
#import jax.numpy as jnp
from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M
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
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       transforms.RandomRotation(90)
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))#,
                       #transforms.RandomRotation(90)
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

test_loader_rot = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       transforms.RandomRotation(90)
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


image_transforms = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ]
)

train_loader_CIFAR = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "../data",
        train=True,
        download=True,
        transform=image_transforms
    ),
    batch_size=64,
    shuffle=True,
)

test_loader_CIFAR = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "../data",
        train=False,
        transform=image_transforms
    ),
    batch_size=1000,
    shuffle=True,
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = P4ConvZ2(1, 10, kernel_size=3)
        self.conv2 = P4ConvP4(10, 10, kernel_size=3)
        self.conv3 = P4ConvP4(10, 20, kernel_size=3)
        self.conv4 = P4ConvP4(20, 20, kernel_size=3)
        self.fc1 = nn.Linear(4*4*20*4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

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
      return F.log_softmax(x)

class AllCnnC(nn.Module):

  def __init__( self, input_size=32*32, input_channels=1, output_size=10):
    """
    :param input_size: number of pixels in the image
    :param input_channels: number of color channels in the image
    :param n_feature: size of the hidden dimensions to use
    :param output_size: expected size of the output
    """
    ksize = 3

    super().__init__()
    self.conv1 = nn.Conv2d(
        in_channels=input_channels, out_channels=96, kernel_size=ksize,padding=1
    )
    self.conv2 = nn.Conv2d(96, 96, kernel_size=ksize,padding=1) #need to add padding to keep the same dimensions of the image 
    self.conv3 = nn.Conv2d(96, 96, kernel_size=ksize,padding=1,stride=2)
    self.conv4 = nn.Conv2d(96, 192, kernel_size=ksize,padding=1)
    self.conv5 = nn.Conv2d(192, 192, kernel_size=ksize,padding=1)
    self.conv6 = nn.Conv2d(192, 192, kernel_size=ksize,padding=1,stride=2)  
    self.conv7 = nn.Conv2d(192,192,kernel_size=ksize,padding=1)
    self.conv8 = nn.Conv2d(192,192,kernel_size=1, padding=1)
    self.conv9 = nn.Conv2d(192,10,kernel_size=1,padding=1)
    self.dropout2d_input = nn.Dropout2d(p=0.2)
    self.dropout2d = nn.Dropout2d(p=0.5)


  def forward(self, 
              x: torch.Tensor
      ) -> torch.Tensor:
      """
      :param x: batch of images with size [batch, 1, w, h]

      :returns: predictions with size [batch, output_size]
      """
      x= self.dropout2d_input(x) #input dropout=0.2
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.dropout2d(self.conv3(x))) #dropout=0.5
      x = F.relu(self.conv4(x))
      x = F.relu(self.conv5(x))
      x = F.relu(self.dropout2d(self.conv6(x))) #dropout=0.5
      x = F.relu(self.conv7(x))
      x = F.relu(self.conv8(x))
      x = F.relu(self.conv9(x))
      x = nn.AvgPool2d((6,6),None,padding=1)(x)
      #x = jnp.squeeze(x, axis=(1,2))
      x = x.view(x.shape[0],-1) 
      return F.log_softmax(x)
    
    
class AllCnnC_p4(nn.Module):

  def __init__( self, input_size=32*32, input_channels=1, output_size=10):
    """
    :param input_size: number of pixels in the image
    :param input_channels: number of color channels in the image
    :param n_feature: size of the hidden dimensions to use
    :param output_size: expected size of the output
    """

    super().__init__()
    self.conv1 = P4ConvZ2(1, 48, kernel_size=3)
    self.conv2 = P4ConvP4(48, 48, kernel_size=3)
    self.conv3 = P4ConvP4(48, 48, kernel_size=3, stride=2)
    self.conv4 = P4ConvP4(48, 96, kernel_size=3)
    self.conv5 = P4ConvP4(96, 96, kernel_size=3)
    self.conv6 = P4ConvP4(96, 96, kernel_size=3, stride=2)
    self.conv7 = P4ConvP4(96, 96, kernel_size=3)
    self.conv8 = P4ConvP4(96, 96, kernel_size=1)
    self.conv9 = P4ConvP4(96, 10, kernel_size=1)

    self.dropout2d_input = nn.Dropout2d(p=0.2)
    self.dropout2d = nn.Dropout2d(p=0.5)


  def forward(self, 
              x: torch.Tensor
      ) -> torch.Tensor:
      """
      :param x: batch of images with size [batch, 1, w, h]

      :returns: predictions with size [batch, output_size]
      """
      x= self.dropout2d_input(x) #input dropout=0.2
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.dropout2d(self.conv3(x))) #dropout=0.5
      x = F.relu(self.conv4(x))
      x = F.relu(self.conv5(x))
      x = F.relu(self.dropout2d(self.conv6(x))) #dropout=0.5
      x = F.relu(self.conv7(x))
      x = F.relu(self.conv8(x))
      x = F.relu(self.conv9(x))
      x = nn.AvgPool2d((6,6),None,padding=1)(x)
      #x = jnp.squeeze(x, axis=(1,2))
      x = x.view(x.shape[0],-1) 
      return F.log_softmax(x)
    
class AllCnnC_p4m(nn.Module):

  def __init__( self, input_size=32*32, input_channels=1, output_size=10):
    """
    :param input_size: number of pixels in the image
    :param input_channels: number of color channels in the image
    :param n_feature: size of the hidden dimensions to use
    :param output_size: expected size of the output
    """

    super().__init__()
    self.conv1 = P4MConvZ2(1, 32, kernel_size=3)
    self.conv2 = P4MConvP4M(32, 32, kernel_size=3)
    self.conv3 = P4MConvP4M(32, 32, kernel_size=3, stride=2)
    self.conv4 = P4MConvP4M(32, 64, kernel_size=3)
    self.conv5 = P4MConvP4M(64, 64, kernel_size=3)
    self.conv6 = P4MConvP4M(64, 64, kernel_size=3, stride=2)
    self.conv7 = P4MConvP4M(64, 64, kernel_size=3)
    self.conv8 = P4MConvP4M(64, 64, kernel_size=1)
    self.conv9 = P4MConvP4M(64, 10, kernel_size=1)

    self.dropout2d_input = nn.Dropout2d(p=0.2)
    self.dropout2d = nn.Dropout2d(p=0.5)


  def forward(self, 
              x: torch.Tensor
      ) -> torch.Tensor:
      """
      :param x: batch of images with size [batch, 1, w, h]

      :returns: predictions with size [batch, output_size]
      """
      x= self.dropout2d_input(x) #input dropout=0.2
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.dropout2d(self.conv3(x))) #dropout=0.5
      x = F.relu(self.conv4(x))
      x = F.relu(self.conv5(x))
      x = F.relu(self.dropout2d(self.conv6(x))) #dropout=0.5
      x = F.relu(self.conv7(x))
      x = F.relu(self.conv8(x))
      x = F.relu(self.conv9(x))
      x = nn.AvgPool2d((6,6),None,padding=1)(x)
      #x = jnp.squeeze(x, axis=(1,2))
      x = x.view(x.shape[0],-1) 
      return F.log_softmax(x)
    
    
class P4CNN_drop(nn.Module):
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
        #x = plane_group_spatial_max_pooling(x, 2, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class P4CNN_no_drop(nn.Module):
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
        #x = plane_group_spatial_max_pooling(x, 2, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class P4CNN_max(nn.Module): #no dropout
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

class P4CNN_RP(nn.Module):
    
  def __init__( self ):
    """
    :param input_size: number of pixels in the image
    :param input_channels: number of color channels in the image
    :param n_feature: size of the hidden dimensions to use
    :param output_size: expected size of the output
    """
    ksize = 3
    ksize_f = 4

    super(P4CNN_RP,self).__init__()
    self.conv1 = P4ConvZ2(
        in_channels=1, out_channels=10, kernel_size=ksize)#,padding=1)
    self.conv2 = P4ConvZ2(10, 10, kernel_size=ksize)#,padding=1) #need to add padding to keep the same dimensions of the image 
    self.conv3 = P4ConvZ2(10, 10, kernel_size=ksize)#,padding=1)
    self.conv4 = P4ConvZ2(10, 10, kernel_size=ksize)#,padding=1)
    self.conv5 = P4ConvZ2(10, 10, kernel_size=ksize)#,padding=1)
    self.conv6 = P4ConvZ2(10, 10, kernel_size=ksize)#,padding=1)
    self.conv7 = P4ConvZ2(10, 10, kernel_size=ksize)#,padding=1)
    self.fc1 = nn.Linear(40, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, 
              x: torch.Tensor
      ) -> torch.Tensor:
      """
      :param x: batch of images with size [batch, 1, w, h]

      :returns: predictions with size [batch, output_size]
      """
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(x,kernel_size=2)# axis=-3, keepdims=False)

      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x,kernel_size=2)# axis=-3, keepdims=False)

      x = plane_group_spatial_max_pooling(x, 2, 2)

      x = F.relu(self.conv3(x))
      x = F.max_pool2d(x, kernel_size=2)# axis=-3, keepdims=False)

      x = F.relu(self.conv4(x))
      x = F.max_pool2d(x,kernel_size=2)# axis=-3, keepdims=False)

      x = F.relu(self.conv5(x))
      x = F.max_pool2d(x,kernel_size=2)#, axis=-3, keepdims=False)

      x = F.relu(self.conv6(x))
      x = F.max_pool2d(x,kernel_size=2)# axis=-3, keepdims=False)

      x = self.conv7(x)
      x = F.max_pool2d(x, kernel_size=2)# axis=-3, keepdims=False)
      x = x.view(x.size()[0], -1)
      x = F.relu(self.fc1(x))
      x = F.dropout(x, training=self.training)
      x = self.fc2(x)
      return F.log_softmax(x)

class P4MCNN(nn.Module): #no dropout
    def __init__(self):
        super(P4MCNN, self).__init__()
        self.conv1 = P4MConvZ2(1, 7, kernel_size=3)
        self.conv2 = P4MConvP4M(7, 7, kernel_size=3)
        self.conv3 = P4MConvP4M(7, 7, kernel_size=3)
        self.conv4 = P4MConvP4M(7, 7, kernel_size=3)
        self.conv5 = P4MConvP4M(7, 7, kernel_size=3)
        self.conv6 = P4MConvP4M(7, 7, kernel_size=3)
        self.conv7 = P4MConvP4M(7, 7, kernel_size=3)
        self.fc1 = nn.Linear(56, 50)
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





#TRIALS 
print('AllCnnC')
model = AllCnnC()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=0.001)

"""
print('AllCNNC')
model = AllCnnC()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
"""
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

def train_cifar(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader_CIFAR):
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
                epoch, batch_idx * len(data), len(train_loader_CIFAR.dataset),
                100. * batch_idx / len(train_loader_CIFAR), loss.data.item()))


def test_cifar():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader_CIFAR:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader_CIFAR.dataset),
        100. * correct / len(test_loader_CIFAR.dataset)))



for epoch in range(1, args.epochs + 1):
    train_cifar(epoch)
    test_cifar()
"""
print('P4CNN drop')
model = P4CNN_drop()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()

print('P4CNN no drop')
model = P4CNN_no_drop()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()

print('P4CNN max')
model = P4CNN_max()
if args.cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()

    """ 
