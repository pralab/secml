"""
.. module:: CNormalizerDNN
   :synopsis: Normalizer which returns the deepfeatures at a specified neural network layer.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Angelo Sotgiu

"""
from secml import _NoValue
from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradientDNNMixin
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

    def fit(self, x, y=None):
        # The inner preprocessor is managed by the inner DNN
        return self._fit(x, y)

    fit.__doc__ = _fit.__doc__  # Same doc of the protected method

    def _transform(self, x):
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
        layer_out = self.net.get_layer_output(x, self.out_layer)
        return layer_out

    def transform(self, x, caching=False):
        self._check_is_fitted()
        # The inner preprocessor is managed by the inner DNN
        return self._transform(x)

    transform.__doc__ = _transform.__doc__  # Same doc of the protected method

    def gradient(self, x, y=None, w=None):
        """Returns the normalizer gradient wrt data.

        Parameters
        ----------
        x : CArray
            Data array, 2-Dimensional or ravel.
        y : int or None, optional
            Index of the class wrt the gradient must be computed.
            This could be not required if w is passed.
        w : CArray or None, optional
            If CArray, will be passed to backward in the net and must have
            a proper shape depending on the chosen output layer.

        Returns
        -------
        gradient : CArray
            Gradient of the normalizer wrt input data. Vector-like array.

        """
        if not issubclass(type(self.net), CClassifierGradientDNNMixin):
            raise NotImplementedError("the DNN wrapper does not implement"
                                      "gradient")
        self._check_is_fitted()

        if not x.is_vector_like:
            raise ValueError('Gradient available only wrt a single point!')

        # For this normalizer we take the net layer output directly
        # So disable the softmax-scaling option
        softmax_outputs = self.net.softmax_outputs
        self.net.softmax_outputs = False

        out_grad = self.net.grad_f_x(
            x, y=y, w=w, layer=self.out_layer)

        # Restore softmax-scaling option
        self.net.softmax_outputs = softmax_outputs

        return out_grad.ravel()
