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
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
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

# CIFAR10 datasets 

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

# All-CNN-C P4M
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
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

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
      #x = nn.AvgPool2d((6,6),None,padding=1)(x)
      #x = jnp.squeeze(x, axis=(1,2))
      x = x.view(x.shape[0],-1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x) 
      return F.log_softmax(x)
    
#Model definition
print('All CNN C p4m')
model = AllCnnC_p4m()
if args.cuda:
    model.cuda()
    
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=0.001)

# Training and testing routines 
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

    test_loss /= len(test_loader_CIFAR.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader_CIFAR.dataset),
        100. * correct / len(test_loader_CIFAR.dataset)))

    
# Training and testing
for epoch in range(1, args.epochs + 1):
    train_cifar(epoch)
    test_cifar()
