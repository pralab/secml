"""
.. module:: SecmlAutograd
    :synopsis: Wraps a secML CModule or chain of CModules inside
    a PyTorch autograd layer.

.. moduleauthor:: Luca Demetrio <luca.demetrio@dibris.unige.it>
.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""
import torch
from torch import nn

from secml.array import CArray


class SecmlAutogradFunction(torch.autograd.Function):
    """
    This class wraps a generic secML classifier inside a PyTorch
    autograd function. When the function's backward is called,
    the secML module calls the internal backward of the CModule,
    and links it to the external graph.
    Reference here:
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, clf, func_call_counter, grad_call_counter):
        ctx.clf = clf
        ctx.save_for_backward(input, grad_call_counter)
        func_call_counter += input.shape[0]
        out = as_tensor(clf.decision_function(as_carray(input)))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        clf = ctx.clf
        input, grad_calls = ctx.saved_tensors
        # https://github.com/pytorch/pytorch/issues/1776#issuecomment-372150869
        with torch.enable_grad():
            grad_input = clf.gradient(x=as_carray(input),
                                      w=as_carray(grad_output))
            grad_calls += clf._cached_x.shape[0]

        grad_input = as_tensor(grad_input, True)
        input_shape = input.shape
        grad_input = grad_input.reshape(input_shape)
        return grad_input, None, None, None


def as_tensor(x, requires_grad=False, tensor_type=None):
    x = torch.from_numpy(x.tondarray().copy()).view(x.input_shape)
    x = x.type(x.dtype if tensor_type is None else tensor_type)
    x.requires_grad = requires_grad
    return x


def as_carray(x, dtype=None):
    return CArray(x.cpu().detach().numpy()).astype(dtype)


class SecmlLayer(nn.Module):
    """
    Defines a PyTorch module that wraps a secml classifier.

    Allows autodiff of the secml modules.

    Credits: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

    Parameters
    ----------
    model : CCLassifier
       Classifier to wrap in the layer. When the layer's backward
       is called, it will internally run the clf's backward and store
       accumulated gradients in the input tensor.
       Function and Gradient call counts will be tracked,
       however they must be reset externally before the call.
    """
    def __init__(self, model):
        super(SecmlLayer, self).__init__()
        self._clf = model
        self.secml_autograd = SecmlAutogradFunction.apply
        self.eval()
        self.func_counter = torch.tensor(0)
        self.grad_counter = torch.tensor(0)

    def forward(self, x):
        x = self.secml_autograd(x, self._clf, self.func_counter,
                                self.grad_counter)
        return x

    def extra_repr(self) -> str:
        return "Wrapper of SecML model {}".format(self._clf)

    def reset(self):
        self.func_counter = torch.tensor(0)
        self.grad_counter = torch.tensor(0)
