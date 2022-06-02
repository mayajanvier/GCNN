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
