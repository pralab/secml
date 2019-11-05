from abc import ABCMeta, abstractmethod
import six


# TODO: use this class as a superclass for CPreProcess, CKernel, CClassifier
@six.add_metaclass(ABCMeta)
class CModule:
    """Common interface for implementing autodiff with forward/backward passes.

    Parameters
    ----------
    preprocess : CPreProcess or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass. If None, input data is used as is.

    """

    def __init__(self, preprocess=None):
        self._preprocess = preprocess
        self._cached_x = None  # cached internal x repr. for backward pass

    @property
    def preprocess(self):
        """Inner preprocessor (if any)."""
        return self._preprocess

    @preprocess.setter
    def preprocess(self, preprocess):
        self._preprocess = preprocess

    def _preprocess_data(self, x, caching=True):
        """Apply the inner preprocess (if any) to the input data x.

        Parameters
        ----------
        x : CArray
            Data to be transformed using inner preprocess, if defined.

        caching: bool
                 True if preprocessed input should be cached for backward pass.

        Returns
        -------
        CArray
            If an inner preprocess is defined, will be the transformed data.
            Otherwise input data is returned as is.

        """
        if self.preprocess is not None:
            # apply preprocessing to x
            x_prc = self.preprocess.forward(x)
        else:
            # use directly x as input to this module
            x_prc = x

        if caching is True:
            # cache intermediate representation of x if required
            self._cached_x = x_prc
        else:
            self._cached_x = None

        return x_prc

    @abstractmethod
    def forward(self, x, caching=True):
        """Forward pass on input x.
        This function internally calls self._preprocess_data(x) to handle
        caching of intermediate representation of the input data x.

        Parameters
        ----------
        x : CArray
            Input array to be transformed.
            Shape of input array depends on the algorithm itself.
        caching: bool
                 True if preprocessed input should be cached for backward pass.

        Returns
        -------
        CArray
            Transformed input data.

        """
        raise NotImplementedError("`forward` not implemented.")

    def backward(self, w=None):
        """Returns the preprocessor gradient wrt data.

        Parameters
        ----------
        w : CArray or None, optional
            If CArray, will be left-multiplied to the gradient
            of the preprocessor.

        Returns
        -------
        gradient : CArray
            Gradient of the preprocessor wrt input data.
        """
        # TODO: handle w = None
        grad = self._backward(w=w)

        if self.preprocess is not None:  # accumulate gradients
            grad = self.preprocess.backward(w=grad)

        return grad.ravel() if grad.is_vector_like else grad

    def _backward(self, w=None):
        raise NotImplementedError("`_backward` is not implemented for {:}"
                                  "".format(self.__class__.__name__))

    _backward.__doc__ = backward.__doc__  # Same doc for the protected method

    def gradient(self, x, w=None):
        """Compute gradient at x by doing a forward and a backward pass.
        The gradient is pre-multiplied by w.
        """
        # TODO: parameters like SVs in kernel.k(x,sv) have to be stored inside
        self.forward(x)
        return self.backward(w)
