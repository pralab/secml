"""
.. module:: CClassifierGradientDNN
   :synopsis: Mixin for DNN classifier gradients.

.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""
from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradientMixin

class CClassifierGradientDNNMixin(CClassifierGradientMixin):
    """Mixin class for CClassifierDNN gradients."""

    def grad_f_x(self, x, y=None, w=None, layer=None):
        """Computes the gradient of the classifier's output wrt input.

        Parameters
        ----------
        x : CArray
            The gradient is computed in the neighborhood of x.
        y : int or None, optional
            Index of the class wrt the gradient must be computed.
            This is not required if:
             - `w` is passed and the last layer is used but
              softmax_outputs is False
             - an intermediate layer is used
        w : CArray or None, optional
            If CArray, will be passed to backward and must have a proper shape
            depending on the chosen output layer (the last one if `layer`
            is None). This is required if `layer` is not None.
        layer : str or None, optional
            Name of the layer.
            If None, the gradient at the last layer will be returned
             and `y` is required if `w` is None or softmax_outputs is True.
            If not None, `w` of proper shape is required.
        **kwargs
            Optional parameters for the function that computes the
            gradient of the decision function. See the description of
            each classifier for a complete list of optional parameters.

        Returns
        -------
        gradient : CArray
            Gradient of the classifier's output wrt input. Vector-like array.

        """
        return CClassifierGradientMixin.grad_f_x(
            self, x=x, y=y, w=w, layer=layer)

    def _grad_f_x(self, x, y=None, w=None, layer=None):
        """Computes the gradient of the classifier's decision function
         wrt input.

        Parameters
        ----------
        x : CArray
            The gradient is computed in the neighborhood of x.
        y : int or None, optional
            Index of the class wrt the gradient must be computed.
            This is not required if:
             - `w` is passed and the last layer is used but
              softmax_outputs is False
             - an intermediate layer is used
        w : CArray or None, optional
            If CArray, will be passed to backward and must have a proper shape
            depending on the chosen output layer (the last one if `layer`
            is None). This is required if `layer` is not None.
        layer : str or None, optional
            Name of the layer.
            If None, the gradient at the last layer will be returned
             and `y` is required if `w` is None or softmax_outputs is True.
            If not None, `w` of proper shape is required.

        Returns
        -------
        gradient : CArray
            Gradient of the classifier's df wrt its input. Vector-like array.

        """
        raise NotImplementedError
