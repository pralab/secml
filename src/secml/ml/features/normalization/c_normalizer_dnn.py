"""
.. module:: CNormalizerDNN
   :synopsis: Normalizer which returns the deepfeatures at a specified neural network layer.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Angelo Sotgiu

"""
from secml import _NoValue
from secml.array import CArray
from secml.ml.features.normalization import CNormalizer
from secml.core.exceptions import NotFittedError


class CNormalizerDNN(CNormalizer):
    """Normalized features are the DNN deepfeatures

    Parameters
    ----------
    net : CClassifierDNN
        DNN to be used for extracting deepfeatures.
        This must be already trained.
    out_layer : str or None, optional
        Identifier of the layer at which the features must be retrieved.
        If None, the output of last layer will be returned.

    Attributes
    ----------
    class_type : 'dnn'

    Notes
    -----
    Any additional inner preprocess should not be passed as the `preprocess`
    parameter but to the DNN instead.

    """
    __class_type = 'dnn'

    def __init__(self, net, out_layer=None, preprocess=_NoValue):

        self._net = net
        self.out_layer = out_layer

        if not self.net.is_fitted():
            raise NotFittedError(
                "the DNN should be already trained.")

        if preprocess is not _NoValue:
            raise ValueError(
                "any additional `preprocess` should be passed to the DNN.")

        # No preprocess should be passed to super
        super(CNormalizerDNN, self).__init__(preprocess=None)

    @property
    def net(self):
        """The DNN."""
        return self._net

    def _check_is_fitted(self):
        """Check if the preprocessor is trained (fitted).

        Raises
        ------
        NotFittedError
            If the preprocessor is not fitted.

        """
        pass  # This preprocessor does not require training

    def _fit(self, x, y=None):
        """Fit normalization algorithm using data.

        This fit function is just a placeholder and simply returns
        the normalizer itself.

        Parameters
        ----------
        x : CArray
            Array to be used for training normalization algorithm.
            Shape of input array depends on the algorithm itself.
        y : CArray or None, optional
            Flat array with the label of each pattern. Not Used.

        Returns
        -------
        CNormalizer
            Instance of the trained normalizer.

        """
        return self

    def fit(self, x, y):
        # The inner preprocessor is managed by the inner DNN
        return self._fit(x, y)

    fit.__doc__ = _fit.__doc__  # Same doc of the protected method

    def _forward(self, x):
        """Apply the transformation algorithm on data.

        This extracts the deepfeatures at the specified layer
        of the DNN.

        Parameters
        ----------
        x : CArray
            Array to be transformed.

        Returns
        -------
        CArray
            Deepfeatures at the specified DNN layer.
            Shape depends on the neural network layer shape.

        """
        return self.net.get_layer_output(x, self.out_layer)

    def _backward(self, w=None):
        # return the gradient at desired layer
        return self.net.get_layer_gradient(x=self._cached_x, w=w,
                                           layer=self.out_layer)
