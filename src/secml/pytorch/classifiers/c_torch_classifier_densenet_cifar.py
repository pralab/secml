"""
.. module:: PyTorchClassifierDenseNet
   :synopsis: Classifier with PyTorch DenseNet Neural Network

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import torch
import torchvision.transforms as transforms

from . import CTorchClassifier
from ..models.cifar import densenet


class CTorchClassifierDenseNetCifar(CTorchClassifier):
    """Torch Classifier with DenseNet CIFAR Neural Network.

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

    Attributes
    ----------
    class_type : 'torch-densenet-cifar'

    """
    __class_type = 'torch-densenet-cifar'

    def __init__(self, batch_size=64, depth=100, growthRate=12, num_classes=10,
                 learning_rate=1e-2, momentum=0.9, weight_decay=1e-4,
                 epochs=300, gamma=0.1, lr_schedule=(150, 225),
                 regularize_bias=True, train_transform=None, preprocess=None):

        # Model params
        self._depth = depth
        self._growthRate = growthRate
        self._num_classes = num_classes

        # Specific parameters of the classifier
        self._classes = None  # TODO: MANAGE LIST OF CLASSES

        super(CTorchClassifierDenseNetCifar, self).__init__(
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            epochs=epochs,
            gamma=gamma,
            lr_schedule=lr_schedule,
            regularize_bias=regularize_bias,
            train_transform=train_transform,
            preprocess=preprocess
        )

    @property
    def depth(self):
        """Model depth."""
        return self._depth

    @property
    def growthRate(self):
        """Growth rate for DenseNet."""
        return self._growthRate

    @property
    def num_classes(self):
        """Number of classes of training dataset."""
        # Wraps our CClassifier.n_classes property as they use num_ as prefix
        return self.n_classes

    @property
    def n_classes(self):
        """Number of classes of training dataset."""
        if self.classes is not None:
            self._num_classes = self.classes.size  # Override the internal param
            return self.classes.size
        else:  # Use the internal parameter
            return self._num_classes

    def _init_model(self):
        """Initialize the PyTorch Neural Network model."""
        self._model = densenet(
            num_classes=self.num_classes,
            depth=self.depth,
            growthRate=self.growthRate,
            compressionRate=2,
            dropRate=0,
        )

    def _get_test_input_loader(self, x, n_jobs=1):
        """Return a loader for input test data."""
        # Convert to CTorchDataset and use a dataloader that returns batches
        dl = super(CTorchClassifierDenseNetCifar, self)._get_test_input_loader(
            x, n_jobs=n_jobs)

        # Add a transformation that reshape samples to (C x H x W)
        dl.dataset.transform = transforms.Lambda(
            lambda p: p.reshape([3, 32, 32]))

        return dl

    def loss(self, x, target):
        """Return the loss function computed on input."""
        # CEL requires targets as long
        return torch.nn.CrossEntropyLoss()(x, torch.max(target, 2)[1].view(-1))
