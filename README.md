# GCNN

Implementation of [Group Equivariant Convolutional Network (T.S. Cohen, M. Wellin, 2016](https://arxiv.org/abs/1602.07576), using PyTorch. 

# Installation 
Install [GrouPy](https://github.com/adambielski/GrouPy), using: 
```
git clone https://github.com/adambielski/GrouPy
cd GrouPy
python setup.py install
```

To run a model, go to the folder with the model you want to try and just run it: 
```
cd my_dataset_folder
python my_model.py
```

# Group Equivariant CNN
## Rotated MNIST 
Implementation of the rotated dataset, baseline Z2CNN, P4CNN with and without dropout, P4CNN Rotation Polling and P4MCNN as an addition to original paper. 
Training routine taken from [Adam Bielski's implementation](https://github.com/adambielski/pytorch-gconv-experiments).


![Rotated one](https://github.com/mayajanvier/GCNN/blob/main/Rotated_MNIST/Rotated%20one.png)


### Baseline model
The baseline for this dataset is a Z2CNN with:
*   7 layers of 3x3 convolutions (4x4 in the final layer), with 20 channels for each layer
*   relu activation functions
*   batch normalization
*   dropout
*   max-pooling after layer 2

Typically, batchnorm is applied before the non-linearity. 

The optimisation is performed with Adam. 

### P4CNN
Implements **p4-convolutions** instead of the classic ones (the group p4 consists of all the compositions of translations and rotations by 90 degrees about any center of rotation in a square grid). A **max-pooling layer over rotations** is also added after the last convolutional layer. 

We need to divide the number of filters by $\sqrt 4=2$ in order to keep the number of parameters approximately fixed: we then have **10 channels** instead of 20. 

Three versions of P4CNN:
*   without the last max pooling layer and with dropout (P4CNN_drop)
*   without the last max pooling layer and without dropout (P4CNN_no_drop)
*   with the last max pooling layer, without dropout (P4CNN_max)

### P4CNN Rotation Pooling
The P4CNN Rotation Pooling is a variant of the former model, but we insert a coset max-pooling layer over rotations. 

### P4MCNN
Same as P4CNN but with **p4m-convolutions** (the group p4m consists of all compositions of translations,
mirror reflections, and rotations by 90 degrees about any center of rotation in the grid).

We need to divide the number of filters by $\sqrt 8\approx 3$ in order to keep the number of parameters approximately fixed: we then have **7 channels** instead of 20.


## CIFAR 10
Implementation of baseline ALL-CNN-C, its p4 modification, baseline ResNet44 and its p4 and p4m modifications. 

### All-CNN-C

Model from [Springenberg et al.(2015)](https://arxiv.org/pdf/1412.6806.pdf), which consists of a sequence of
9 strided and non-strided convolution layers, interspersed with rectified linear activation units. 

![all-cnn-c.JPG](https://github.com/mayajanvier/GCNN/blob/main/allcnnc.JPG) 


### All-CNN-C p4 and p4m
Derived from All-CNN-C, replacing each convolutional layer by its p4 or p4m version. 

# Group Equivariant GAN
Using group equivariant convolutional networks to create a GAN.

## Vanilla GAN
Baseline Vanilla GAN from [Sovit Ranjan Rath](https://debuggercafe.com/generating-mnist-digit-images-using-vanilla-gan-with-pytorch/) on Rotated MNIST. 

## P4 Vanilla GAN
The discriminator is replaced by P4CNN_no_drop. 

