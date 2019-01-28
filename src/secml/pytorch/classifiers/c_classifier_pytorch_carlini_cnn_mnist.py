"""
.. module:: CClassifierPyTorchCarliniCNNMNIST
   :synopsis: Classifier with a CNN for the MNIST datset

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""

from secml.pytorch.classifiers import CClassifierPyTorch
from secml.pytorch.models.mnist import carlini_mnist_model


class CClassifierPyTorchCarliniCNNMNIST(CClassifierPyTorch):
    """

    Loss function for training: cross-entropy (includes softmax)

    Parameters
    ----------
    num_classes : int, optional
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
        Size of the batch for grouping samples. Default 64.
    regularize_bias : bool, optional
        If False, L2 regularization will NOT be applied to biases.
        Default True, so regularization will be applied to all parameters.
        If weight_decay is 0, regularization will not be applied anyway.
        If fit.warm_start is True, this parameter has no effect.
    train_transform : torchvision.transform or None, optional
        Transformation to be applied before training.
    preprocess : CNormalizer or None, optional
        Preprocessing for data. If not None and model state will be loaded
        using `.load_state()`, this should be an already-trained preprocessor
        or `.preprocess.fit(x)` should be called after `.load_state(x)`
        with appropriate input.
    softmax_outputs : bool, optional
        If True, apply softmax function to the outputs. Default False.
    random_state : int or None, optional
        If int, random_state is the seed used by the random number generator.
        If None, no fixed seed will be set.

    Attributes
    ----------
    class_type : 'pytorch-carlini-cnn-mnist'

    """
    __class_type = 'pytorch-carlini-cnn-mnist'

    # def get_lr(epoch):
    #   return base_lr * (.5 ** (epoch / num_epochs * 10))

    def __init__(self, num_classes=10, learning_rate=0.1, momentum=0.9,
                 weight_decay=0, epochs=30, gamma=0.1, batch_size=128,
                 regularize_bias=True, lr_schedule=(1000,),
                 train_transform=None, preprocess=None, softmax_outputs=False,
                 random_state=None):
        super(CClassifierPyTorchCarliniCNNMNIST, self).__init__(
            model=carlini_mnist_model,
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
            input_shape=(28, 28, 1),
            softmax_outputs=softmax_outputs,
            random_state=random_state,
            num_classes=num_classes
        )
