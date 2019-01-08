import torch
from collections import OrderedDict

from . import CTorchClassifier
from secml.utils.dict_utils import merge_dicts


# FIXME: UPDATE CLASS DOCSTRING
class CTorchClassifierFullyConnected(CTorchClassifier):
    """Torch classifier with Fully-Connected Neural Network.

    Fully-connected neural network with two hidden layers and
     ReLU as activation function.

    Parameters
    ----------
    input_dims : int, optional
        Size of the input layer. Default 1000.
    hidden_dims : int, optional
        Size of the hidden layers. Default 100.
    output_dims : int, optional
        Size of the output layer. Default 10.
    learning_rate : float, optional
        Learning rate. Default 1e-2.
    momentum : float, optional
        Momentum factor. Default 0.9.
    weight_decay : float, optional
        Weight decay (L2 penalty). Control parameters regularization.
        Default 1e-4.
    n_epoch : int, optional
        Number of epochs. Default 100.
    gamma : float, optional
        Multiplicative factor of learning rate decay. Default: 0.1.
    lr_schedule : tuple, optional
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
    class_type : 'torch-fc'

    """
    __class_type = 'torch-fc'

    def __init__(self,  batch_size=5, input_dims=1000, hidden_dims=100,
                 output_dims=10, learning_rate=1e-2, momentum=0.9,
                 weight_decay=1e-4, n_epoch=100, gamma=0.1,
                 lr_schedule=(50, 75), regularize_bias=True,
                 train_transform=None, preprocess=None):

        # Specific parameters of the classifier
        self._input_dims = input_dims
        self._hidden_dims = hidden_dims
        self._output_dims = output_dims

        super(CTorchClassifierFullyConnected, self).__init__(
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            n_epoch=n_epoch,
            gamma=gamma,
            lr_schedule=lr_schedule,
            regularize_bias=regularize_bias,
            train_transform=train_transform,
            preprocess=preprocess
        )

        self._init_params = merge_dicts(self._init_params,
                                        {'input_dims': input_dims,
                                         'hidden_dims': hidden_dims,
                                         'output_dims': output_dims})

    def _init_model(self):
        """Initialize the PyTorch Neural Network model."""
        # Use the nn package to define our model as a sequence of layers.
        # nn.Sequential is a Module which contains other Modules, and applies
        # them in sequence to produce its output. Each Linear Module computes
        # output from input using a linear function, and holds internal
        # Tensors for its weight and bias. After constructing the model
        # we use the .to() method to move it to the desired device
        self._model = torch.nn.Sequential(OrderedDict([
            ('linear1', torch.nn.Linear(self._input_dims, self._hidden_dims)),
            ('relu1', torch.nn.ReLU()),
            ('linear2', torch.nn.Linear(self._hidden_dims, self._hidden_dims)),
            ('relu2', torch.nn.ReLU()),
            ('linear3', torch.nn.Linear(self._hidden_dims, self._output_dims)),
            ('softmax', torch.nn.Softmax(dim=-1))
        ]))

    def loss(self, x, target):
        """Return the loss function computed on input."""
        # The nn package also contains definitions of popular loss functions;
        # in this case we will use Mean Squared Error (MSE) as our loss
        return torch.nn.MSELoss(size_average=False)(x, target.float())
