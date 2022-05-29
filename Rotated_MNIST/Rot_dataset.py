""" Visualization of a rotated image from the train dataset"""

import plotly.express as px
from Rotated_MNIST.train import train_loader_MNIST

mnist_example = next(iter(train_loader_MNIST))
fig = px.imshow(mnist_example[0][1][0])
fig.show()
