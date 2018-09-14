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


class CTorchClassifierDenseNet(CTorchClassifier):
    """"""
    class_type = 'nn_densenet'

    def __init__(self, depth=100, growthRate=12, num_classes=10,
                 learning_rate=1e-2, momentum=0.9, weight_decay=1e-4,
                 n_epoch=300, gamma=0.1, lr_schedule=(150, 225), batch_size=64,
                 train_transform=None, normalizer=None):

        # Specific parameters of the classifier
        self._num_classes = num_classes
        self._depth = depth
        self._growthRate = growthRate

        test_transform = transforms.Compose(
            [transforms.Lambda(lambda x: x.reshape([3, 32, 32]))])

        super(CTorchClassifierDenseNet, self).__init__(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            n_epoch=n_epoch,
            gamma=gamma,
            lr_schedule=lr_schedule,
            batch_size=batch_size,
            train_transform=train_transform,
            test_transform=test_transform,
            normalizer=normalizer
        )

        self._init_params = merge_dicts(self._init_params,
                                        {'num_classes': num_classes,
                                         'depth': depth,
                                         'growthRate': growthRate})

    def _init_model(self):
        """Initialize the PyTorch Neural Network model."""
        self._model = densenet(
            num_classes=self._num_classes,
            depth=self._depth,
            growthRate=self._growthRate,
            compressionRate=2,
            dropRate=0,
        )

    def loss(self, x, target):
        """Return the loss function computed on input."""
        # CEL requires targets as long
        return torch.nn.CrossEntropyLoss()(x, torch.max(target, 2)[1].view(-1))
