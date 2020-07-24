"""
.. module:: CClassifierPyTorch
   :synopsis: Generic wrapper for PyTorch classifiers.

.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""
from functools import reduce

import torch
from torch import nn
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
    for name, layer in net._modules.items():
        # If it is a sequential, don't return its name
        # but recursively register all it's module children
        if isinstance(layer, nn.Sequential) or isinstance(layer, BasicBlock):
            yield from [(":".join([name, l]), m) for (l, m) in get_layers(layer)]
        else:
            yield (name, layer)


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
        number of workers to use for data loading and processing.
        This parameter follows the library expected behavior of having 1 worker
        as the main process. The loader will spawn `n_jobs-1` workers.
        Default value for n_jobs is 1 (zero additional workers spawned).

    Attributes
    ----------
    class_type : 'pytorch-clf'

    """
    __class_type = 'pytorch-clf'

    def __init__(self, model, loss=None,
                 optimizer=None,
                 optimizer_scheduler=None,
                 pretrained=False,
                 pretrained_classes=None,
                 input_shape=None,
                 random_state=None, preprocess=None,
                 softmax_outputs=False,
                 epochs=10, batch_size=1, n_jobs=1):

        self._device = self._set_device()
        self._random_state = random_state

        super(CClassifierPyTorch, self).__init__(
            model=model,
            preprocess=preprocess,
            pretrained=pretrained,
            pretrained_classes=pretrained_classes,
            input_shape=input_shape,
            softmax_outputs=softmax_outputs, n_jobs=n_jobs)

        self._init_model()
        self._batch_size = batch_size

        if self._batch_size is None:
            self.logger.info(
                "No batch size passed. Value will be set to the default "
                "value of 1.")
            self._batch_size = 1

        if self._input_shape is None:
            # try to infer from first layer
            first_layer = list(self._model._modules.values())[0]
            if isinstance(first_layer, torch.nn.Linear):
                self._input_shape = (first_layer.in_features,)
            else:
                raise ValueError(
                    "Input shape should be specified if the first "
                    "layer is not a `nn.Linear` module.")

        self._loss = loss
        self._optimizer = optimizer
        self._optimizer_scheduler = optimizer_scheduler

        self._epochs = epochs

        if self._pretrained is True:
            self._trained = True
            if self._pretrained_classes is not None:
                self._classes = self._pretrained_classes
            else:
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

    @loss.setter
    def loss(self, loss):
        """Sets the loss function to use for training."""
        self._loss = loss

    @property
    def model(self):
        """Returns the model used by classifier."""
        return self._model

    @property
    def optimizer(self):
        """Returns the optimizer used by classifier."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Sets the optimizer for the DNN."""
        self._optimizer = optimizer

    @property
    def optimizer_scheduler(self):
        """Returns the optimizer used by classifier."""
        return self._optimizer_scheduler

    @optimizer_scheduler.setter
    def optimizer_scheduler(self, optimizer_scheduler):
        """Sets the scheduler for training the DNN"""
        self._optimizer_scheduler = optimizer_scheduler

    @property
    def epochs(self):
        """Returns the number of epochs for which the model
        will be trained."""
        return self._epochs

    @epochs.setter
    def epochs(self, epochs):
        """Sets the number of epochs for training."""
        self._epochs = epochs

    @property
    def batch_size(self):
        """Returns the batch size used for the dataset loader."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        """Sets the batch size for loading the batches."""
        self._batch_size = batch_size

    @property
    def layers(self):
        """Returns the layers of the model, if possible. """
        if self._model_layers is None:
            if isinstance(self._model, nn.Module):
                self._model_layers = list(get_layers(self._model))
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

    def get_params(self):
        """Returns the dictionary of class parameters."""
        loss_params = {'loss': self._loss}
        optim_params = {
            'optimizer':
                self._optimizer.state_dict()['param_groups'][0]
                if self._optimizer is not None else None,
            'optimizer_scheduler':
                self._optimizer_scheduler.state_dict()
                if self._optimizer_scheduler is not None else None
        }
        return SubLevelsDict(
            merge_dicts(super(CClassifierPyTorch, self).get_params(),
                        loss_params, optim_params))

    def get_state(self, return_optimizer=True):
        """Returns the object state dictionary.

        Parameters
        ----------
        return_optimizer : bool, optional
            If True (default), state of `optimizer` and `optimizer_scheduler`,
            if defined, will be included in the state dictionary.

        Returns
        -------
        dict
            Dictionary containing the state of the object.

        """
        from copy import deepcopy

        # State of the wrapping classifier
        state = super(CClassifierPyTorch, self).get_state()

        # Map model to CPU before saving
        self._model.to(torch.device('cpu'))

        # Use deepcopy as restoring device later will change them
        state['model'] = deepcopy(self._model.state_dict())

        # Restore device for model
        self._model.to(self._device)

        # When `return_optimizer` is False we do not include `optimizer` and
        # `optimizer_scheduler` in state dict. However, if `return_optimizer`
        # is True, `optimizer` and `optimizer_scheduler` should be included,
        # even if they are None
        if return_optimizer is False:
            state.pop('optimizer')
            state.pop('optimizer_scheduler')
        else:
            # Unfortunately optimizer does not have a 'to(device)' method
            if self._optimizer is not None:
                for opt_state in self._optimizer.state.values():
                    for k, v in opt_state.items():
                        if isinstance(v, torch.Tensor):
                            opt_state[k] = v.to('cpu')

                # Use deepcopy as restoring device later will change them
                state['optimizer'] = deepcopy(self._optimizer.state_dict())

                # Restore optimizer state to proper device
                for opt_state in self._optimizer.state.values():
                    for k, v in opt_state.items():
                        if isinstance(v, torch.Tensor):
                            opt_state[k] = v.to(self._device)

                if self._optimizer_scheduler is not None:
                    # Scheduler will be saved only if also optimizer is defined
                    # No need to map to `cpu`, tensors in state
                    state['optimizer_scheduler'] = deepcopy(
                        self._optimizer_scheduler.state_dict())

        return state

    def set_state(self, state_dict, copy=False):
        """Sets the object state using input dictionary."""
        # TODO: DEEPCOPY FOR torch.load_state_dict?

        self._model.load_state_dict(state_dict.pop('model'))

        if 'optimizer' in state_dict:
            if self._optimizer is None:
                raise ValueError(
                    "optimizer not found in current object but required for "
                    "restoring state."
                    "Save the state using `return_optimizer=False` or "
                    "add an optimizer to the model first.")
            self._optimizer.load_state_dict(state_dict.pop('optimizer'))

        if 'optimizer_scheduler' in state_dict:
            if self._optimizer_scheduler is None:
                raise ValueError(
                    "`optimizer_scheduler` not found in current object "
                    "but required for restoring state."
                    "Save the state using `return_optimizer=False` or "
                    "add an optimizer scheduler to the model first.")
            self._optimizer_scheduler.load_state_dict(
                state_dict.pop('optimizer_scheduler'))

        super(CClassifierPyTorch, self).set_state(state_dict, copy=copy)

    def __getattribute__(self, key):
        """Get an attribute.

        This allows getting also attributes of the internal PyTorch model,
        loss and optimizer."""
        try:
            # If we are not getting the model itself
            if key not in ['_model', '_optimizer', '_optimizer_scheduler']:
                if hasattr(self, '_model') and key in self._model._modules:
                    return self._model[key]
                elif hasattr(self, '_optimizer') and \
                        self._optimizer is not None and \
                        key in self._optimizer.state_dict()['param_groups'][0]:
                    if len(self._optimizer.state_dict()['param_groups']) == 1:
                        return self._optimizer.param_groups[0][key]
                    else:
                        raise NotImplementedError(
                            "__getattribute__ is not yet supported for "
                            "optimizers with more than one element in "
                            "param_groups.")
                elif hasattr(self, '_optimizer_scheduler') and \
                        self._optimizer_scheduler is not None and \
                        key in self._optimizer_scheduler.state_dict():
                    return self._optimizer_scheduler[key]

        except (KeyError, AttributeError):
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
        elif hasattr(self, '_optimizer_scheduler') and \
                self._optimizer_scheduler is not None and \
                key in self._optimizer_scheduler.state_dict():
            self._optimizer_scheduler.state_dict[key] = value
        else:  # Otherwise, normal python set behavior
            super(CClassifierPyTorch, self).__setattr__(key, value)

    def _init_model(self):
        """Initialize the PyTorch Neural Network model."""
        # Setting random seed
        if self._random_state is not None:
            torch.manual_seed(self._random_state)
            torch.backends.cudnn.deterministic = True

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

    def _data_loader(self, data, labels=None, batch_size=10,
                     shuffle=False, num_workers=0):
        """Returns `torch.DataLoader` generated from the input CDataset.

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
            Number of additional processes to use for loading the data.
            Default value is 0.

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

    def _fit(self, x, y):
        """Fit PyTorch model.

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features).
        y : CArray
            Array of shape (n_samples,) containing the class labels.

        """
        if any([self._optimizer is None,
                self._loss is None]):
            raise ValueError("Optimizer and loss should both be defined "
                             "in order to fit the model.")

        train_loader = self._data_loader(
            x, y, batch_size=self._batch_size, num_workers=self.n_jobs - 1)

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

            if self._optimizer_scheduler is not None:
                self._optimizer_scheduler.step()

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
        data_loader = self._data_loader(x, num_workers=self.n_jobs - 1,
                                        batch_size=self._batch_size)

        # Switch to evaluation mode
        self._model.eval()

        out_shape = self.n_classes if self._out_layer is None else \
            reduce((lambda z, v: z * v), self.layer_shapes[self._out_layer])
        output = torch.empty((len(data_loader.dataset), out_shape))

        for batch_idx, (s, _) in enumerate(data_loader):
            # Log progress
            self.logger.info(
                'Classification: {batch}/{size}'.format(batch=batch_idx,
                                                        size=len(data_loader)))

            s = s.to(self._device)

            if self._cached_x is None:
                self._cached_s = None
                self._cached_layer_output = None
                with torch.no_grad():
                    ps = self._get_layer_output(s, self._out_layer)

            else:
                # keep track of the gradient in s tensor
                s.requires_grad = True
                ps = self._get_layer_output(s, self._out_layer)
                self._cached_s = s
                self._cached_layer_output = ps

            output[batch_idx * self.batch_size:
                   batch_idx * self.batch_size + len(s)] = \
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
        if w is None:
            raise ValueError("Function `_backward` needs the `w` array "
                             "to run backward with.")

        # Apply softmax-scaling if needed (only if last layer is required)
        if self.softmax_outputs is True and self._out_layer is None:
            out_carray = self._from_tensor(
                self._cached_layer_output.squeeze(0).data)
            softmax_grad = CArray.zeros(shape=out_carray.shape[0])
            for y in w.nnz_indices[1]:
                softmax_grad += w[y] * CSoftmax().gradient(
                    out_carray, y=y)
            w = softmax_grad

        w = self._to_tensor(w.atleast_2d()).reshape(
            self._cached_layer_output.shape)
        w = w.to(self._device)

        if self._cached_s.grad is not None:
            self._cached_s.grad.data._zero()

        self._cached_layer_output.backward(w)

        return self._from_tensor(self._cached_s.grad.data.view(
            -1, reduce(lambda a, b: a * b, self.input_shape)))

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
            'n_features': self.n_features,
            'classes': self.classes,
        }

        if self.optimizer is not None:
            state['optimizer_state'] = self._optimizer.state_dict()

        if self._optimizer_scheduler is not None:
            state['optimizer_scheduler_state'] = \
                self._optimizer_scheduler.state_dict()

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
        keys = ['model_state', 'n_features', 'classes']
        if all(key in state for key in keys):
            if classes is not None:
                self.logger.warning(
                    "Model was saved within `secml` framework. "
                    "The parameter `classes` will be ignored.")
            # model was stored with save_model method
            self._model.load_state_dict(state['model_state'])

            if 'optimizer_state' in state \
                    and self._optimizer is not None:
                self._optimizer.load_state_dict(state['optimizer_state'])
            else:
                self._optimizer = None

            if 'optimizer_scheduler_state' in state \
                    and self._optimizer_scheduler is not None:
                self._optimizer_scheduler.load_state_dict(
                    state['optimizer_scheduler_state'])
            else:
                self._optimizer_scheduler = None

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
