"""
.. module:: PytorchModelUtils
   :synopsis: Collection of utilities for PyTorch models

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import torch.nn as nn
import torch.nn.init as init


def init_params(net):
    """Initialize layers parameters.

    Supported layers:
     - nn.Conv2d
     - nn.BatchNorm2d
     - nn.Linear

    Parameters
    ----------
    net : torch.nn.Module

    Notes
    -----
    Credits to https://github.com/bearpaw/pytorch-classification

    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)
