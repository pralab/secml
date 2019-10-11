"""
.. module:: CClassifierPyTorch
   :synopsis: Generic wrapper for PyTorch classifiers.

.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""
from functools import reduce
from collections import OrderedDict

import torch
from torch import nn
import torchvision
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import transforms

from secml.array import CArray
from secml.data.loader import CDataLoaderPyTorch
from secml.ml.classifiers import CClassifier
from secml.ml.classifiers.gradients import CClassifierGradientPyTorchMixin
from secml.utils import SubLevelsDict, merge_dicts

from secml.settings import SECML_PYTORCH_USE_CUDA

use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA


def get_layers(net):
    for name, layer in net._modules.items():
        # If it is a sequential, don't return its name
        # but recursively register all it's module children
        if isinstance(layer, nn.Sequential) or isinstance(layer, BasicBlock):
                yield from [(":".join([name, l]), m) for (l, m) in get_layers(layer)]
        else:
            yield (name, layer)


class CClassifierPyTorch(CClassifier, CClassifierGradientPyTorchMixin):
    """Generic wrapper for PyTorch model."""
    __class_type = 'pytorch-clf'

    def __init__(self, torch_model, loss=None, optimizer=None,
                 input_shape=None,
                 random_state=None, preprocess=None,
                 softmax_outputs=False,
                 epochs=10, batch_size=1, n_jobs=1):
        """
        CClassifierPyTorch
        Wrapper for PyTorch models.

        Parameters
        ----------
        torch_model:
            model to use
        loss:
            loss
        optimizer:

        random_state: int or None, optional
            random state to use for initializing the model weights.
            Default value is None.
        preprocess:
            preprocessing module.
        softmax_outputs: bool, optional
            if set to True, a softmax function will be applied to
            the return value of the decision function. Note: some
            implementation adds the softmax function to the network
            class as last layer or last forward function, or even in the
            loss function (see torch.nn.CrossEntropyLoss). Be aware that the
            softmax may have already been applied.
            Default value is False.
        epochs: int
            number of epochs for training the neural network. Default value is 10.
        batch_size: int
            size of the batches to use for loading the data. Default value is 1.
        n_jobs: int
            number of workers to use for data loading and processing. Default value is 1.

        Attributes
        ----------
        class_type : 'pytorch-clf'

        """

        super(CClassifierPyTorch, self).__init__(preprocess=preprocess)

        self._device = torch.device("cuda" if use_cuda else "cpu")
        self._model = torch_model
        self._random_state = random_state
        self._init_model()
        self._trained = False

        self._input_shape = input_shape
        if self._input_shape is None:
            # try to infer from first layer
            first_layer = list(self._model._modules.values())[0]
            if isinstance(first_layer, torch.nn.Linear):
                self._input_shape = (first_layer.in_features,)
            else:
                raise ValueError("Input shape should be specified if the first "
                                 "layer is not a `nn.Linear` module.")

        # check softmax redundancy
        if isinstance(loss, nn.CrossEntropyLoss) and self.check_softmax():
            raise ValueError("Please remove softmax redundancy. Either "
                             "use `torch.nn.NLLLoss` or remove softmax "
                             "layer from the network.")
        self._loss = loss
        self._optimizer = optimizer

        if self._optimizer is not None:
            # check softmax redundancy
            if self.check_softmax() and softmax_outputs:
                self.logger.warning("Softmax layer has been defined in the network. Disabling "
                                    "parameter softmax_outputs.")
                self._softmax_outputs = False
            else:
                self._softmax_outputs = softmax_outputs
        else:
            self._softmax_outputs = False

        self._epochs = epochs
        self._batch_size = batch_size

        if self._batch_size is None:
            self.logger.info("No batch size passed. Value will be set to the default "
                             "value of 1.")
            self._batch_size = 1

        self._n_jobs = n_jobs

        if self._model.__class__.__name__ in dir(torchvision.models):
            self._trained = True
            self._classes = CArray.arange(list(self._model.modules())[-1].out_features)
            self._n_features = reduce(lambda a, b: a * b, self._input_shape)

        # hooks for getting intermediate outputs
        self._handlers = []
        # will store intermediate outputs from the hooks
        self._intermediate_outputs = None

    @property
    def loss(self):
        """Returns the loss function used by classifier."""
        return self._loss

    @property
    def optimizer(self):
        """Returns the optimizer used by classifier."""
        return self._optimizer

    @property
    def epochs(self):
        """Returns the number of epochs for which the model
        will be trained."""
        return self._epochs

    @property
    def device(self):
        """Returns the device that is being used for model and data."""
        return self._device

    @property
    def batch_size(self):
        """Returns the batch size used for the dataset loader."""
        return self._batch_size

    @property
    def input_shape(self):
        """Returns the input shape of the first layer of the neural network."""
        return self._input_shape

    @input_shape.setter
    def input_shape(self, input_shape):
        self._input_shape = input_shape

    @property
    def softmax_outputs(self):
        return self._softmax_outputs

    @softmax_outputs.setter
    def softmax_outputs(self, active):
        """

        Parameters
        ----------
        active : bool
            whether apply softmax to the final scores

        Notes
        ----------
        If the loss is an instance of `nn.CrossEntropyLoss` or
        the network already has a softmax operation in the end
        this parameter will be disabled.
        """
        self.check_softmax()
        self._softmax_outputs = active

    @property
    def layers(self):
        """Returns the layers of the model, if possible. """
        if isinstance(self._model, nn.Module):
            return get_layers(self._model)
        else:
            raise TypeError("The input model must be an instance of `nn.Module`.")

    @property
    def layer_names(self):
        return list(zip(*self.layers))[0]

    def get_layer_shape(self, layer_name):
        layer = next(filter(lambda x: x[0] == layer_name, get_layers(self._model)))[1]
        if isinstance(layer, nn.Linear):
            return (1, layer.out_features)
        else:
            raise ValueError("Layer output shape can be requested only for linear layers.")

    def _clean_hooks(self):
        """Removes previously defined hooks."""
        for handler in self._handlers:
            handler.remove()

    def _hook_forward(self, module_name, input, output):
        """Hooks the module's `forward` method so that it stores
        the intermediate outputs as tensors."""

        self._intermediate_outputs[module_name] = output

    def hook_layer_output(self, layer_names=None):
        """
        Creates handlers for the hooks that store the layer outputs.

        Parameters
        ----------
        layer_names : list
            List of layer names to hook. Cleans previously
            defined hooks to prevent multiple hook creations.

        """

        self._clean_hooks()
        self._handlers = []
        self._intermediate_outputs = {}

        for name, layer in get_layers(self._model):
            if name in layer_names:
                self._handlers.append(layer.register_forward_hook(self._hook_forward))
            else:
                pass

    def n_jobs(self):
        """Returns the number of workers being used for loading
        and processing the data."""
        return self._n_jobs

    def get_params(self):
        loss_params = {'loss': self._loss}
        optim_params = {
            'optimizer': self._optimizer.state_dict()['param_groups'][0] if self._optimizer is not None else None}
        return SubLevelsDict(
            merge_dicts(super(CClassifierPyTorch, self).get_params(),
                        loss_params, optim_params))

    def check_softmax(self):
        """
        Checks if a softmax layer has been defined in the
        network.

        Returns
        -------
        Boolean value stating if a softmax layer has been
        defined.
        """
        x = torch.ones(tuple([1] + list(self.input_shape)))
        x = x.to(self._device)

        outputs = self._model(x)

        if outputs.sum() == 1:
            return True
        return False

    def __getattribute__(self, key):
        """Get an attribute.

        This allows getting also attributes of the internal PyTorch model,
        loss and optimizer."""
        try:
            # If we are not getting the model itself
            if key not in ['_model', '_optimizer']:
                if hasattr(self, '_model') and key in self._model._modules:
                    return self._model[key]
                elif hasattr(self, '_optimizer') and \
                        self._optimizer is not None and \
                        key in self._optimizer.state_dict()['param_groups'][0]:
                    if len(self._optimizer.state_dict()['param_groups']) == 1:
                        return self._optimizer.param_groups[0][key]
                    else:
                        raise NotImplementedError("__getattribute__ is not yet "
                                                  "supported for optimizers with "
                                                  "more than one element in "
                                                  "param_groups.")
        except KeyError:
            pass  # Parameter not found in PyTorch model
            # Try to get the parameter from self
        return super(CClassifierPyTorch, self).__getattribute__(key)

    def __setattr__(self, key, value):
        """Set an attribute.

        This allow setting also the attributes of the internal PyTorch model.

        """
        if isinstance(value, (torch.Tensor, torch.nn.Module)):
            value = value.to(self._device)
        if hasattr(self, '_model') and key in self._model._modules:
            self._model._modules[key] = value
        elif hasattr(self, '_optimizer') and \
                self._optimizer is not None and \
                key in self._optimizer.state_dict()['param_groups'][0]:
            self._optimizer.param_groups[0][key] = value
        else:  # Otherwise, normal python set behavior
            super(CClassifierPyTorch, self).__setattr__(key, value)

    def _init_model(self):
        """Initialize the PyTorch Neural Network model."""
        # Setting random seed
        if self._random_state is not None:
            torch.manual_seed(self._random_state)

        # Make sure that model is a proper PyTorch module
        if not isinstance(self._model, nn.Module):
            raise TypeError("`model` must be a `torch.nn.Module`.")

        self._model = self._model.to(self._device)

    @staticmethod
    def _to_tensor(x):
        """Convert input CArray to tensor."""
        if not isinstance(x, CArray):
            raise ValueError("A `CArray` is required as "
                             "input to the `_to_tensor` method.")
        x = x.tondarray()
        x = torch.from_numpy(x)
        x = x.type(torch.FloatTensor)
        if use_cuda is True:
            x = x.cuda(device=torch.device('cuda'))
        return x

    @staticmethod
    def _from_tensor(x):
        """Convert input tensor to CArray"""
        if not isinstance(x, torch.Tensor):
            raise ValueError("A `torch.Tensor` is required as "
                             "input to the `_from_tensor` method.")
        return CArray(x.cpu().numpy()).astype(float)

    def _data_loader(self, data, labels=None, batch_size=10, shuffle=False, num_workers=1):
        """
        Returns `torch.DataLoader` generated from
        the input CDataset.

        Parameters
        ----------
        data : CArray
            CArray containing the input data to load.
        labels : CArray
            CArray containing the labels for the data.
        batch_size : int, optional
            Size of the batches to load for each iter of
            the data loader.
            Default value is 10.
        shuffle : bool, optional
            Whether to shuffle the data before dividing in batches.
            Default value is False.
        num_workers : int, optional
            Number of processes to use for loading the data.
            Default value is 1.

        Returns
        -------
        `CDataLoaderPyTorch` iterator for loading the dataset in batches,
        optionally shuffled, with the specified number of workers.

        """
        transform = transforms.Lambda(lambda x: x.reshape(self._input_shape))
        return CDataLoaderPyTorch(data, labels,
                                  batch_size, shuffle=shuffle,
                                  transform=transform,
                                  num_workers=num_workers, ).get_loader()

    def _fit(self, dataset):
        """Fit PyTorch model."""

        if any([self._optimizer is None,
                self._loss is None]):
            raise ValueError("Optimizer and loss should both be defined "
                             "in order to fit the model.")

        train_loader = self._data_loader(dataset.X, dataset.Y,
                                         batch_size=self._batch_size)

        for epoch in range(self._epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                self._optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self._loss(outputs, labels)
                loss.backward()
                self._optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        self._trained = True
        return self._model

    def predict(self, x, return_decision_function=False):
        """

        Parameters
        ----------
        x : CArray
            Array with samples to classify, of shape (n_patterns, n_features).
        return_decision_function : bool, optional
            if True, returns the decision_function value along
            with the predictions. Default is False.

        Returns
        -------
        labels : CArray
            Flat dense array of shape (n_patterns,) with the label assigned
             to each test pattern. The classification label is the label of
             the class associated with the highest score.
        scores : CArray, optional
            Array of shape (n_patterns, n_classes) with classification
             score of each test pattern with respect to each training class.
            Will be returned only if `return_decision_function` is True.

        """
        self._check_is_fitted()

        scores = self._decision_function(x)

        # The classification label is the label of the class
        # associated with the highest score
        labels = scores.argmax(axis=1)

        return (labels, scores) if return_decision_function is True else labels

    def _decision_function(self, x, y=None):
        """Implementation of the decision function."""

        x_carray = CArray(x).atleast_2d()

        # Transform data if a preprocess is defined
        x_carray = self._preprocess_data(x_carray)

        data_loader = self._data_loader(x_carray, num_workers=self._n_jobs,
                                        batch_size=self._batch_size)

        # Switch to evaluation mode
        self._model.eval()

        output = torch.empty((len(data_loader.dataset), self.n_classes))

        for batch_idx, (s, _) in enumerate(data_loader):
            # Log progress
            self.logger.info(
                'Classification: {batch}/{size}'.format(batch=batch_idx,
                                                        size=len(data_loader)))

            s = s.to(self._device)

            with torch.no_grad():
                ps = self._model(s)
                ps = ps.squeeze(1)

            output[batch_idx * len(s):(batch_idx + 1) * len(s)] = ps

        # Apply softmax-scaling if needed
        if self._softmax_outputs is True:
            scores = output.softmax(dim=1)
        else:
            scores = output

        scores = self._from_tensor(scores)
        return scores

    def get_layer_output(self, x, layer_names=None):
        """Returns the output of the desired net layer as `CArray`.

        Parameters
        ----------
        x : CArray
            Input data.
        layer_names : str, list or None, optional
            Name of the layer(s) to hook for getting the outputs.
            If None, the output of the last layer will be returned.

        Returns
        -------
        CArray or dict
            Output of the desired layers, dictionary if more than one layer is
            requested.

        """
        self._check_is_fitted()

        x = CArray(x).atleast_2d()

        # Transform data if a preprocess is defined
        x = self._preprocess_data(x)

        x, _ = next(iter(self._data_loader(x, None)))
        x = x.to(self._device)

        with torch.no_grad():
            # handle single layer name passed
            if isinstance(layer_names, str):
                layer_names = [layer_names]

            # Get the model output at specific layer
            out = self._get_layer_output(x, layer_names=layer_names)

            if isinstance(out, dict):
                out = {k: self._from_tensor(v.view(-1)) for (k, v) in out.items()}
            else:
                out = self._from_tensor(out)

        return out

    def _get_layer_output(self, s, layer_names=None):
        """Returns the output of the desired net layer as `Torch.Tensor`.

        Parameters
        ----------
        s : torch.Tensor
            Input tensor to forward propagate.
        layer_names : list or None, optional
            Name of the layer(s) to hook for getting the output.
            If None, the output of the last layer will be returned.

        Returns
        -------
        torch.Tensor
            Output of the desired layer.

        """
        # Switch to evaluation mode
        self._model.eval()
        if layer_names is None:  # Directly use the last layer
            return self._model(s)  # Forward pass

        elif isinstance(layer_names, list) or isinstance(layer_names, str):
            if isinstance(layer_names, str):
                layer_names = [layer_names]

            self.hook_layer_output(layer_names)
            self._model(s)
            return {layer_names[i]: v for i, (k, v) in enumerate(self._intermediate_outputs.items())}
        else:
            raise ValueError("Pass layer names as a list or just None for last layer output.")

    def save_checkpoint(self):
        pass

    def save_model(self, filename):
        """
        Stores the model and optimizer's parameters.

        Parameters
        ----------
        filename : str
            path of the file for storing the model

        """
        state = {
            'model_state': self._model.state_dict(),
            'optimizer_state': self._optimizer.state_dict(),
            'n_features': self.n_features,
            'classes': self.classes,
        }

        torch.save(state, filename)

    def load_model(self, filename):
        """
        Restores the model and optimizer's parameters.
        Notes: the model class and optimizer should be
        defined before loading the params.

        Parameters
        ----------
        filename : str
            path where to find the stored model

        """
        state = torch.load(filename, map_location=self._device)

        self._model.load_state_dict(state['model_state'])
        self._optimizer.load_state_dict(state['optimizer_state'])
        self._n_features = state['n_features']
        self._classes = state['classes']
