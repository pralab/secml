"""
.. module:: CNNMNIST
   :synopsis: A CNN model for MNIST

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from collections import OrderedDict

import torch.nn as nn

__all__ = ['cnn_mnist_model']


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return x


class CNNMNIST(nn.Module):
    """
    The CNN learned on the MNIST dataset by Carlini in the paper:
    N. Carlini and D. A. Wagner, "Adversarial examples are not easily
    detected: Bypassing ten detection methods"

    Original code:
    - https://github.com/carlini/nn_breaking_detection/blob/master/setup_mnist.py

    Parameters
    ----------
    num_classes : int
        Number of classes (outputs). Default 10.
    init_strategy : str
        If 'default', use the default initialization strategy
         for all the layers,
        If `fan_out`, use the default init strategy for the
         linear layer and the `kaiming_normal` init strategy with
         the option fan_out for the convolutional layers.
        If `uniform_scaling` use the uniform scaling strategy
         for all the layers

    """

    def __init__(self, num_classes=10, init_strategy='default'):

        nb_filters = 64

        super(CNNMNIST, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, nb_filters, kernel_size=5, stride=2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=nb_filters,
                                out_channels=nb_filters,
                                kernel_size=3, stride=2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=nb_filters,
                                out_channels=nb_filters,
                                kernel_size=3, stride=1)),
            ('relu3', nn.ReLU(inplace=True)),
            ('flatten', Flatten()),
            ('fc1', nn.Linear(576, out_features=32)),
            ('relu4', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout(p=.5, inplace=False)),
            ('flatten2', Flatten()),
        ]))

        self.classifier = nn.Linear(32, num_classes)

        if init_strategy == "fan_out":
            # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')
        elif init_strategy == 'uniform_scaling':
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    uniform_scaling_(m.weight)
                    uniform_scaling_(m.bias)
        elif init_strategy == "default":
            # Delving deep into rectifiers: Surpassing human - level
            # performance on ImageNet classification - He, K. et al. (2015)
            pass
        else:
            raise ValueError("Unknown initialization strategy!")

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def cnn_mnist_model(**kwargs):
    model = CNNMNIST(**kwargs)
    return model


def uniform_scaling_(tensor, factor=1.0):
    r"""Initialization with random values from uniform distribution without scaling
    variance.

    When initializing a deep network, it is in principle advantageous to keep
    the scale of the input variance constant, so it does not explode or diminish
    by reaching the final layer. If the input is `x` and the operation `x * W`,
    and we want to initialize `W` uniformly at random, we need to pick `W` from

      [-sqrt(3) / sqrt(dim), sqrt(3) / sqrt(dim)]

    to keep the scale intact, where `dim = W.shape[0]` (the size of the input).
    A similar calculation for convolutional networks gives an analogous result
    with `dim` equal to the product of the first 3 dimensions.  When
    nonlinearities are present, we need to multiply this by a constant `factor`.
    See [Sussillo et al., 2014](https://arxiv.org/abs/1412.6558)
    ([pdf](http://arxiv.org/pdf/1412.6558.pdf)) for deeper motivation, experiments
    and the calculation of constants. In section 2.3 there, the constants were
    numerically computed: for a linear layer it's 1.0, relu: ~1.43, tanh: ~1.15.

    Arguments:
        factor: `float`. A multiplicative factor by which the values will be
            scaled.
        dtype: The tensor data type. Only float are supported.
        seed: `int`. Used to create a random seed for the distribution.

    Returns:
        The Initializer, or an initialized `Tensor` if shape is specified.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution

    """
    import torch
    import math
    with torch.no_grad():
        shape = tensor.shape
        input_size = 1.0
        for dim in shape[:-1]:
            input_size *= float(dim)
        max_val = math.sqrt(3 / input_size) * factor
        return torch.FloatTensor(shape).uniform_(-max_val, max_val)
