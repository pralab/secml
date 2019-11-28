"""
.. module:: CClassifierPyTorch
   :synopsis: Generic wrapper for PyTorch classifiers.

.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""
from functools import reduce

import torch
from torch import nn
import torchvision
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import transforms

from secml.array import CArray
from secml.data.loader import CDataLoaderPyTorch
from secml.ml.classifiers import CClassifierDNN
from secml.ml.classifiers.loss import CSoftmax
from secml.utils import SubLevelsDict, merge_dicts
from secml.ml.classifiers.gradients import CClassifierGradientMixin

from secml.settings import SECML_PYTORCH_USE_CUDA

use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA


def get_layers(net):
    # TODO remove when dropping support for python 2
    layers = list()
    for name, layer in net._modules.items():
        # If it is a sequential, don't return its name
        # but recursively register all it's module children
        if isinstance(layer, nn.Sequential) or isinstance(layer, BasicBlock):
            layers += [(":".join([name, l]), m) for (l, m) in
                       get_layers(layer)]
        else:
            layers.append((name, layer))
    else:
        return layers

    # TODO and uncomment this
    # for name, layer in net._modules.items():
    #     # If it is a sequential, don't return its name
    #     # but recursively register all it's module children
    #     if isinstance(layer, nn.Sequential) or isinstance(layer, BasicBlock):
    #         yield from [(":".join([name, l]), m) for (l, m) in get_layers(layer)]
    #     else:
    #         yield (name, layer)


class CClassifierPyTorch(CClassifierDNN, CClassifierGradientMixin):
    """CClassifierPyTorch, wrapper for PyTorch models.

    Parameters
    ----------
    model:
        `torch.nn.Module` object to use as classifier
    loss:
        loss object from `torch.nn`
    optimizer:
        optimizer object from `torch.optim`

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
    __class_type = 'pytorch-clf'

    def __init__(self, model, loss=None, optimizer=None,
                 input_shape=None,
                 random_state=None, preprocess=None,
                 softmax_outputs=False,
                 epochs=10, batch_size=1, n_jobs=1):

        self._device = self._set_device()
        self._random_state = random_state
        super(CClassifierPyTorch, self).__init__(model=model,
                                                 preprocess=preprocess,
                                                 input_shape=input_shape,
                                                 softmax_outputs=softmax_outputs)
        self._init_model()

        if self._input_shape is None:
            # try to infer from first layer
            first_layer = list(self._model._modules.values())[0]
            if isinstance(first_layer, torch.nn.Linear):
                self._input_shape = (first_layer.in_features,)
            else:
                raise ValueError(
                    "Input shape should be specified if the first "
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
                self.logger.warning(
                    "Softmax layer has been defined in the network. Disabling "
                    "parameter softmax_outputs.")
                self._softmax_outputs = False
            else:
                self._softmax_outputs = softmax_outputs
        else:
            self._softmax_outputs = False

        self._epochs = epochs
        self._batch_size = batch_size

        if self._batch_size is None:
            self.logger.info(
                "No batch size passed. Value will be set to the default "
                "value of 1.")
            self._batch_size = 1

        self._n_jobs = n_jobs

        if self._model.__class__.__name__ in dir(torchvision.models):
            self._trained = True
            self._classes = CArray.arange(
                list(self._model.modules())[-1].out_features)
            self._n_features = reduce(lambda a, b: a * b, self._input_shape)

        # hooks for getting intermediate outputs
        self._handlers = []
        # will store intermediate outputs from the hooks
        self._intermediate_outputs = None
        self._cached_s = None
        self._cached_layer_output = None

    @property
    def loss(self):
        """Returns the loss function used by classifier."""
        return self._loss

    @property
    def model(self):
        """Returns the model used by classifier."""
        return self._model

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
    def batch_size(self):
        """Returns the batch size used for the dataset loader."""
        return self._batch_size

    @property
    def layers(self):
        """Returns the layers of the model, if possible. """
        if self._model_layers is None:
            if isinstance(self._model, nn.Module):
                self._model_layers = get_layers(self._model)
            else:
                raise TypeError(
                    "The input model must be an instance of `nn.Module`.")
        return self._model_layers

    @property
    def layer_shapes(self):
        if self._model_layer_shapes is None:
            self._model_layer_shapes = {}
            layer_names = self.layer_names
            self.hook_layer_output(layer_names)
            x = torch.randn(size=self.input_shape).unsqueeze(0)
            x = x.to(self._device)
            self._model(x)
            for layer_name, layer in self.layers:
                self._model_layer_shapes[layer_name] = tuple(
                    self._intermediate_outputs[layer].shape)
            self._clean_hooks()
        return self._model_layer_shapes

    @property
    def trained(self):
        """True if the model has been trained."""
        return self._trained

    def get_layer_shape(self, layer_name):
        return self.layer_shapes[layer_name]

    def _clean_hooks(self):
        """Removes previously defined hooks."""
        for handler in self._handlers:
            handler.remove()
        self._intermediate_outputs = None

    def _hook_forward(self, module_name, input, output):
        """Hooks the module's `forward` method so that it stores
        the intermediate outputs as tensors."""

        self._intermediate_outputs[module_name] = output

    def hook_layer_output(self, layer_names=None):
        """
        Creates handlers for the hooks that store the layer outputs.

        Parameters
        ----------
        layer_names : list or str, optional
            List of layer names to hook. Cleans previously
            defined hooks to prevent multiple hook creations.

        """
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        self._clean_hooks()
        self._handlers = []
        self._intermediate_outputs = {}

        for name, layer in get_layers(self._model):
            if name in layer_names:
                self._handlers.append(
                    layer.register_forward_hook(self._hook_forward))
            else:
                pass

    def _set_device(self):
        return torch.device("cuda" if use_cuda else "cpu")

    def n_jobs(self):
        """Returns the number of workers being used for loading
        and processing the data."""
        return self._n_jobs

    def get_params(self):
        """Returns the dictionary of class parameters."""
        loss_params = {'loss': self._loss}
        optim_params = {
            'optimizer': self._optimizer.state_dict()['param_groups'][0]
            if self._optimizer is not None else None
        }
        return SubLevelsDict(
            merge_dicts(super(CClassifierPyTorch, self).get_params(),
                        loss_params, optim_params))

    def get_state(self):
        """Returns the object state dictionary."""
        state = super(CClassifierPyTorch, self).get_state()
        state['model'] = self._model.state_dict()
        state['optimizer'] = self._optimizer.state_dict()
        return state

    def set_state(self, state_dict, copy=False):
        """Sets the object state using input dictionary."""
        # TODO: DEEPCOPY FOR torch.load_state_dict?
        self._model.load_state_dict(state_dict.pop('model'))
        self._optimizer.load_state_dict(state_dict.pop('optimizer'))
        super(CClassifierPyTorch, self).set_state(state_dict, copy=copy)

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
                        raise NotImplementedError(
                            "__getattribute__ is not yet "
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

    def _data_loader(self, data, labels=None, batch_size=10, shuffle=False,
                     num_workers=1):
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
                    self.logger.info('[%d, %5d] loss: %.3f' %
                                     (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        self._trained = True
        return self._model

    def _forward(self, x):
        """Forward pass on input x.
        Returns the output of the layer set in _out_layer.
        If _out_layer is None, the last layer output is returned,
        after applying softmax if softmax_outputs is True.

        Parameters
        ----------
        x : CArray
            preprocessed array, ready to be transformed by the current module.

        Returns
        -------
        CArray
            Transformed input data.

        """
        data_loader = self._data_loader(x, num_workers=self._n_jobs,
                                        batch_size=self._batch_size)

        # Switch to evaluation mode
        self._model.eval()

        out_shape = self.n_classes if self._out_layer is None else \
            reduce((lambda x, y: x * y), self.layer_shapes[self._out_layer])
        output = torch.empty((len(data_loader.dataset), out_shape))

        for batch_idx, (s, _) in enumerate(data_loader):
            # Log progress
            self.logger.info(
                'Classification: {batch}/{size}'.format(batch=batch_idx,
                                                        size=len(data_loader)))

            s = s.to(self._device)

            if self._cached_x is None:
                with torch.no_grad():
                    ps = self._get_layer_output(s, self._out_layer)

            else:
                # keep track of the gradient in s tensor
                s.requires_grad = True
                ps = self._get_layer_output(s, self._out_layer)
                self._cached_s = s
                self._cached_layer_output = ps

            output[batch_idx * len(s):(batch_idx + 1) * len(s)] = \
                ps.view(ps.size(0), -1).detach()

        # Apply softmax-scaling if needed
        if self._softmax_outputs is True and self._out_layer is None:
            scores = output.softmax(dim=1)
        else:
            scores = output

        scores = self._from_tensor(scores)
        return scores

    def _get_layer_output(self, s, layer_name=None):
        """Returns the output of the desired net layer as `Torch.Tensor`.

        Parameters
        ----------
        s : torch.Tensor
            Input tensor to forward propagate.
        layer_name : str or None, optional
            Name of the layer to hook for getting the output.
            If None, the output of the last layer will be returned.

        Returns
        -------
        torch.Tensor
            Output of the desired layer(s).

        """
        if layer_name is None:  # Directly use the last layer
            return self._model(s)  # Forward pass

        elif isinstance(layer_name, str):

            self.hook_layer_output(layer_name)
            self._model(s)

            if not self._intermediate_outputs:
                raise ValueError("None of requested layers were found")

            return list(self._intermediate_outputs.values())[0]
        else:
            raise ValueError("Pass layer names as a list or just None "
                             "for last layer output.")

    def _backward(self, w):
        """Returns the gradient of the DNN - considering the output layer set
        in _out_layer - wrt data.

        Parameters
        ----------
        w : CArray
            Weights that are pre-multiplied to the gradient
            of the module, as in standard reverse-mode autodiff.

        Returns
        -------
        gradient : CArray
            Accumulated gradient of the module wrt input data.
        """
        if w is not None:
            w = self._to_tensor(w.atleast_2d()).reshape(
                self._cached_layer_output.shape)
        else:
            raise ValueError(
                "Function `_backward` needs the `w` array to run backward with.")

        # Apply softmax-scaling if needed (only if last layer is required)
        if self.softmax_outputs is True and self._out_layer is None:
            out_carray = self._from_tensor(
                self._cached_layer_output.squeeze(0).data)
            softmax_grad = CSoftmax().gradient(
                out_carray, y=self._cached_layer_output)
            self._cached_layer_output *= self._to_tensor(
                softmax_grad.atleast_2d()).unsqueeze(0)
        w = w.to(self._device)

        if self._cached_s.grad is not None:
            self._cached_s.grad.data._zero()

        self._cached_layer_output.backward(w)

        return self._from_tensor(self._cached_s.grad.data.view(
            -1, self.n_features))

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

    def load_model(self, filename, classes=None):
        """
        Restores the model and optimizer's parameters.
        Notes: the model class and optimizer should be
        defined before loading the params.

        Parameters
        ----------
        filename : str
            path where to find the stored model
        classes : list, tuple or None, optional
            This parameter is used only if the model was stored
            with native PyTorch.
            Class labels (sorted) for matching classes to indexes
            in the loaded model. If classes is None, the classes
            will be assigned new indexes from 0 to n_classes.

        """
        state = torch.load(filename, map_location=self._device)
        keys = ['model_state', 'optimizer_state', 'n_features', 'classes']
        if all(key in state for key in keys):
            if classes is not None:
                self.logger.warning(
                    "Model was saved within `secml` framework. "
                    "The parameter `classes` will be ignored.")
            # model was stored with save_model method
            self._model.load_state_dict(state['model_state'])
            self._optimizer.load_state_dict(state['optimizer_state'])
            self._n_features = state['n_features']
            self._classes = state['classes']
        else:  # model was stored outside secml framework
            try:
                self._model.load_state_dict(state)
                # This part is important to prevent not fitted
                if classes is None:
                    self._classes = CArray.arange(
                        self.layer_shapes[self.layer_names[-1]][1])
                else:
                    self._classes = CArray(classes)
                self._n_features = reduce(lambda x, y: x * y, self.input_shape)
                self._trained = True
            except Exception:
                self.logger.error(
                    "Model's state dict should be stored according to "
                    "PyTorch docs. Use `torch.save(model.state_dict())`.")
