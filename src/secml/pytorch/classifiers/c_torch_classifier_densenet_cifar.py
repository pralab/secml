"""
.. module:: PyTorchClassifierDenseNet
   :synopsis: Classifier with PyTorch DenseNet Neural Network

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import torch
import torchvision.transforms as transforms

from . import CTorchClassifier
from ..models.cifar import densenet
from secml.utils.dict_utils import merge_dicts


class CTorchClassifierDenseNetCifar(CTorchClassifier):
    """"""
    class_type = 'nn_densenet'

    def __init__(self, depth=100, growthRate=12, num_classes=10,
                 learning_rate=1e-2, momentum=0.9, weight_decay=1e-4,
                 n_epoch=300, gamma=0.1, lr_schedule=(150, 225), batch_size=64,
                 train_transform=None, normalizer=None):

        # Specific parameters of the classifier
        self._classes = None  # TODO: MANAGE LIST OF CLASSES
        self._n_classes = num_classes
        self._depth = depth
        self._growthRate = growthRate

        super(CTorchClassifierDenseNetCifar, self).__init__(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            n_epoch=n_epoch,
            gamma=gamma,
            lr_schedule=lr_schedule,
            batch_size=batch_size,
            train_transform=train_transform,
            normalizer=normalizer
        )

        self._init_params = merge_dicts(self._init_params,
                                        {'num_classes': num_classes,
                                         'depth': depth,
                                         'growthRate': growthRate})

    @property
    def n_classes(self):
        """Number of classes of training dataset."""
        if self.classes is not None:
            self._n_classes = self.classes.size  # Override the internal param
            return self.classes.size
        else:  # Use the internal parameter
            return self._n_classes

    def _init_model(self):
        """Initialize the PyTorch Neural Network model."""
        self._model = densenet(
            num_classes=self.n_classes,
            depth=self._depth,
            growthRate=self._growthRate,
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
