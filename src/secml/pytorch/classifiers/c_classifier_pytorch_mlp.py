"""
.. module:: PyTorchClassifierMLP
   :synopsis: Classifier with PyTorch Multi-layer Perceptron

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from . import CClassifierPyTorch
from ..models import mlp


class CClassifierPyTorchMLP(CClassifierPyTorch):
    """PyTorch Multi-layer Perceptron Classifier.

    Multi-layer Perceptron neural network with ReLU as activation function.

    Loss function for training: cross-entropy (includes softmax)

    Parameters
    ----------
    input_dims : int, optional
        Size of the input layer. Default 1000.
    hidden_dims : tuple, optional
        Size of the hidden layers. Each value in the tuple represents
        an hidden layer. Default (100, 100), so a network with
        two hidden layers and 100 neurons each.
    output_dims : int, optional
        Size of the output layer. Default 10.
    learning_rate : float, optional
        Learning rate. Default 1e-2.
    momentum : float, optional
        Momentum factor. Default 0.9.
    weight_decay : float, optional
        Weight decay (L2 penalty). Control parameters regularization.
        Default 1e-4.
    epochs : int, optional
        Number of epochs. Default 100.
    gamma : float, optional
        Multiplicative factor of learning rate decay. Default: 0.1.
    lr_schedule : list, optional
        List of epoch indices. Must be increasing.
        The current learning rate will be multiplied by gamma
        once the number of epochs reaches each index.
    batch_size : int, optional
        Size of the batch for grouping samples. Default 5.
    regularize_bias : bool, optional
        If False, L2 regularization will NOT be applied to biases.
        Default True, so regularization will be applied to all parameters.
        If weight_decay is 0, regularization will not be applied anyway.
        If fit.warm_start is True, this parameter has no effect.
    train_transform : torchvision.transform or None, optional
        Transformation to be applied before training.
    preprocess : CNormalizer or None, optional
        Preprocessing for data.
    softmax_outputs : bool, optional
        If True, apply softmax function to the outputs. Default True.
    random_state : int or None, optional
        If int, random_state is the seed used by the random number generator.
        If None, no fixed seed will be set.

    Attributes
    ----------
    class_type : 'pytorch-mlp'

    """
    __class_type = 'pytorch-mlp'

    def __init__(self, input_dims=1000, hidden_dims=(100, 100), output_dims=10,
                 learning_rate=1e-2, momentum=0.9, weight_decay=1e-4,
                 epochs=100, gamma=0.1, lr_schedule=(50, 75),
                 batch_size=5,  regularize_bias=True, train_transform=None,
                 preprocess=None, softmax_outputs=True, random_state=None):

        super(CClassifierPyTorchMLP, self).__init__(
            model=mlp,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            loss='cross-entropy',
            epochs=epochs,
            gamma=gamma,
            lr_schedule=lr_schedule,
            batch_size=batch_size,
            regularize_bias=regularize_bias,
            train_transform=train_transform,
            preprocess=preprocess,
            input_shape=(1, input_dims),
            softmax_outputs=softmax_outputs,
            random_state=random_state,
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
        )
