"""
.. module:: PyTorchClassifierDenseNet
   :synopsis: Classifier with PyTorch DenseNet Neural Network

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from . import CClassifierPyTorch
from ..models.cifar import densenet


class CClassifierPyTorchDenseNetCifar(CClassifierPyTorch):
    """PyTorch Classifier with DenseNet CIFAR Neural Network.

    Loss function for training: cross-entropy (includes softmax)

    Parameters
    ----------
    depth : int, optional
        Model depth. Default 100.
    growthRate : int, optional
        Growth rate for DenseNet. Default 12.
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
        Maximum number of epochs of the training process. Default 100.
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
    preprocess : CPreProcess or str or None, optional
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
    class_type : 'pytorch-densenet-cifar'

    """
    __class_type = 'pytorch-densenet-cifar'

    def __init__(self, depth=100, growthRate=12, compressionRate=2,
                 dropRate=0, num_classes=10, learning_rate=1e-2, momentum=0.9,
                 weight_decay=1e-4, epochs=300,  gamma=0.1,
                 lr_schedule=(150, 225), batch_size=64, regularize_bias=True,
                 train_transform=None, preprocess=None, softmax_outputs=False,
                 random_state=None):

        super(CClassifierPyTorchDenseNetCifar, self).__init__(
            model=densenet,
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
            input_shape=(3, 32, 32),
            softmax_outputs=softmax_outputs,
            random_state=random_state,
            depth=depth,
            growthRate=growthRate,
            compressionRate=compressionRate,
            dropRate=dropRate,
            num_classes=num_classes
        )
