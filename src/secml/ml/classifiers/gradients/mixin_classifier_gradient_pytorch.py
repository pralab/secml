"""
.. module:: CClassifierGradientPytorchMixin
   :synopsis: Mixin for Pytorch classifier gradients.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""
import torch

from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradientMixin
from secml.ml.classifiers.loss import CSoftmax

from secml.settings import SECML_PYTORCH_USE_CUDA
use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA


class CClassifierGradientPyTorchMixin(CClassifierGradientMixin):
    """Mixin class for CClassifierPyTorch gradients."""

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
        if x.is_vector_like is False:
            raise ValueError("gradient can be computed on one sample only.")

        # Transform data if a preprocess is defined
        s, _ = next(iter(self._data_loader(x)))

        if use_cuda is True:
            s = s.cuda()

        # keep track of the gradient in s tensor
        s.requires_grad = True

        # Get the model output at specific layer
        out = self._get_layer_output(s, layer=layer)

        # unsqueeze if net output does not take into account the batch size
        if len(out.shape) < len(s.shape):
            out = out.unsqueeze(0)

        if w is None:
            if layer is not None:
                raise ValueError(
                    "grad can be implicitly created only for the last layer. "
                    "`w` is needed when `layer` is not None.")
            if y is None:  # if layer is None -> y is required
                raise ValueError("The class label wrt compute the gradient "
                                 "at the last layer is required.")

            w_in = torch.FloatTensor(1, out.shape[-1])
            if use_cuda is True:
                w_in = w_in.cuda()
            w_in.zero_()
            w_in[0, y] = 1  # create a mask to get the gradient wrt y
        else:
            w_in = self._to_tensor(w.atleast_2d())

        w_in = w_in.unsqueeze(0)  # unsqueeze to simulate a single point batch

        # Apply softmax-scaling if needed
        if layer is None and self.softmax_outputs is True:
            out_carray = self._from_tensor(out.squeeze(0).data)
            softmax_grad = CSoftmax().gradient(out_carray, y=y)
            w_in *= self._to_tensor(softmax_grad.atleast_2d()).unsqueeze(0)
        elif w is not None and y is not None:
            # Inform the user y has not been used
            self.logger.warning("`y` will be ignored!")

        if s.grad is not None:
            s.grad.data._zero()

        out.backward(w_in)  # Backward on `out` (grad will appear on `s`

        return self._from_tensor(s.grad.data.view(-1))
