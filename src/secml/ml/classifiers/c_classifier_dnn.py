"""
.. module:: CClassifierDNN
   :synopsis: Base class for defining a DNN backend.

.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""
from abc import ABCMeta, abstractmethod

import six

from secml.array import CArray
from secml.ml.classifiers import CClassifier


@six.add_metaclass(ABCMeta)
class CClassifierDNN(CClassifier):
    """Generic wrapper for DNN model."""
    __class_type = ' dnn-clf'

    def __init__(self, model, input_shape=None, preprocess=None,
                 softmax_outputs=False, **kwargs):
        """
        CClassifierDNN
        Wrapper for DNN models.

        Parameters
        ----------
        model:
            backend-supported model
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

        Attributes
        ----------
        class_type : 'dnn-clf'

        """

        super(CClassifierDNN, self).__init__(preprocess=preprocess)

        self._model = model
        self._trained = False
        self._input_shape = input_shape
        self._softmax_outputs = softmax_outputs
        self._layers = None
        self._layer_shapes = None

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
        Defines whether to apply softmax to the final scores.

        Parameters
        ----------
        active : bool
            Activates the softmax in the output

        Notes
        ----------
        If the loss has softmax layer defined, or
        the network already has a softmax operation in the end
        this parameter will be disabled.
        """
        self.check_softmax()
        self._softmax_outputs = active

    @property
    @abstractmethod
    def layers(self):
        """Returns list of tuples containing the layers of the model.
        Each tuple is structured as (layer_name, layer)."""
        raise NotImplementedError

    @property
    def layer_names(self):
        """Returns the names of the layers of the model."""
        return list(zip(*self.layers))[0]

    @property
    @abstractmethod
    def layer_shapes(self):
        """Returns a dictionary containing the shapes of the output
        of each layer of the model."""
        raise NotImplementedError

    def get_params(self):
        """

        Returns
        -------

        """
        return super(CClassifierDNN, self).get_params()

    @abstractmethod
    def check_softmax(self):
        """
        Checks if a softmax layer has been defined in the
        network.

        Returns
        -------
        Boolean value stating if a softmax layer has been
        defined.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _to_tensor(x):
        """Convert input CArray to backend-supported tensor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _from_tensor(x):
        """Convert input backend-supported tensor to CArray"""
        raise NotImplementedError

    @abstractmethod
    def _fit(self, dataset):
        """Fit the model."""

        raise NotImplementedError

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

    @abstractmethod
    def _decision_function(self, x, y=None):
        """
        Computes the output scores of the last layer.
        If `self.softmax_outputs` is True, applies softmax scaling to the
        outputs.

        Parameters
        ----------
        x : CArray
            Array of input samples
        y : CArray, optional

        Returns
        -------

        scores: CArray
            Array of shape (n_patterns, n_classes) with classification
            score of each test pattern with respect to each training class.

        """
        raise NotImplementedError

    @abstractmethod
    def get_layer_output(self, x, layer_names=None):
        """Returns the output of the desired net layer(s).

        Parameters
        ----------
        x : CArray
            Input data.
        layer_names : str, list or None, optional
            Name of the layer(s) to get the output from.
            If None, the output of the last layer will be returned.

        Returns
        -------
        CArray or dict
            Output of the desired layers, dictionary if more than one layer is
            requested.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_layer_output(self, s, layer_names=None):
        """Returns the output of the desired net layer(s) as backend-supported
        tensor.

        Parameters
        ----------
        s : backend-supported tensor
            Input tensor to forward propagate.
        layer : str or None, optional
            Name of the layer.
            If None, the output of the last layer will be returned.

        Returns
        -------
        dict
            Output of the desired layers (as backend-supported
            tensors), dictionary if more than one layer is
            requested.

        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, filename):
        """
        Stores the model and optimization parameters.

        Parameters
        ----------
        filename : str
            path of the file for storing the model

        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, filename):
        """
        Restores the model and optimization parameters.
        Notes: the model class should be
        defined before loading the params.

        Parameters
        ----------
        filename : str
            path where to find the stored model

        """
        raise NotImplementedError
