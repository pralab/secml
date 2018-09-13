import torch
from collections import OrderedDict

from . import CTorchClassifier
from prlib.utils.dict_utils import merge_dicts


class CTorchClassifierFullyConnected(CTorchClassifier):
    """
    A fully-connected ReLU network with one hidden layer, trained to predict y from x
    by minimizing squared Euclidean distance.
    This implementation uses the nn package from PyTorch to build the network.
    PyTorch autograd makes it easy to define computational graphs and take gradients,
    but raw autograd can be a bit too low-level for defining complex neural networks;
    this is where the nn package can help. The nn package defines a set of Modules,
    which you can think of as a neural network layer that has produces output from
    input and may have some trainable weights or other state.
    """
    class_type = 'nn_mlp'

    def __init__(self, input_dims=1000, hidden_dims=100, output_dims=10,
                 learning_rate=1e-2, momentum=0.9, weight_decay=1e-4,
                 n_epoch=100, gamma=0.1, lr_schedule=(50, 75), batch_size=5,
                 train_transform=None, test_transform=None, normalizer=None):

        # Specific parameters of the classifier
        self._input_dims = input_dims
        self._hidden_dims = hidden_dims
        self._output_dims = output_dims

        super(CTorchClassifierFullyConnected, self).__init__(
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
