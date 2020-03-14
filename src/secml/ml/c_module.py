"""
.. module:: CModule
   :synopsis: Common interface for implementing pre-processing chains and
    automatic differentiation with forward/backward passes.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Angelo Sotgiu <angelo.sotgiu@unica.it>

"""
from abc import ABCMeta, abstractmethod
from secml.core import CCreator


class CModule(CCreator, metaclass=ABCMeta):
    """Common interface for handling pre-processing chains and implementing
     automatic differentiation with forward/backward passes.

    Parameters
    ----------
    preprocess : CPreProcess or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass. If None, input data is used as is.

    """
    __super__ = 'CModule'

    def __init__(self, preprocess=None):
        self._preprocess = preprocess
        self._cached_x = None  # cached internal x repr. for backward pass

    @abstractmethod
    def _check_is_fitted(self):
        """Check if the module is trained (fitted).

        Raises
        ------
        NotFittedError
            If the module is not fitted.

        """
        raise NotImplementedError

    def _check_input(self, x):
        """Check if input is properly formatted

        Raises
        ------
        ValueError
            if x is not properly formatted.

        """
        # TODO: make abstract and raise exception.
        #  at this stage we pass as no checks are implemented by default
        # raise NotImplementedError
        pass

    @property
    def preprocess(self):
        """Inner preprocessor (if any)."""
        return self._preprocess

    @preprocess.setter
    def preprocess(self, preprocess):
        self._preprocess = preprocess

    def _preprocess_data(self, x, caching=True):
        """This function prepares the input for `_forward` and `_backward`.

        It checks if x has the proper format and the current module is fitted.
        Then, it applies inner pre-processing (if defined), and caches
        the input data for backward pass (when required).

        Parameters
        ----------
        x : CArray
            Input data to be transformed via pre-processing, if set.

        caching: bool
            True if preprocessed input should be cached for backward pass.

        Returns
        -------
        CArray
            Either the input data x (if pre-processing is not set), or its
            transformed version after pre-processing.

        """
        self._cached_x = None  # reset cached values (if any)

        x = x.atleast_2d()  # Ensuring input is 2-D
        self._check_input(x)
        self._check_is_fitted()

        if self.preprocess is not None:
            # apply preprocessing to x
            x_prc = self.preprocess.forward(x)
        else:
            # use directly x as input to this module
            x_prc = x

        if caching is True:
            # cache intermediate representation of x if required,
            # e.g., if backward has to be called after forward.
            self._cached_x = x_prc

        return x_prc

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
        # Transform data using inner preprocess, if defined
        x = self._preprocess_data(x, caching=caching)
        return self._forward(x)

    @abstractmethod
    def _forward(self, x):
        """Forward pass on input x.

        Parameters
        ----------
        x : CArray
            preprocessed array, ready to be transformed by the current module.

        Returns
        -------
        CArray
            Transformed input data.

        """
        raise NotImplementedError("`_forward` not implemented.")

    def backward(self, w=None):
        """Returns the preprocessor gradient wrt data.

        Parameters
        ----------
        w : CArray or None
            if CArray, it is pre-multiplied to the gradient
            of the module, as in standard reverse-mode autodiff.

        Returns
        -------
        gradient : CArray
            Accumulated gradient of the module wrt input data.

        """

        if self._cached_x is None:
            raise ValueError("Please run forward with caching=True first.")

        grad = self._backward(w=w)

        if self.preprocess is not None:  # accumulate gradients
            grad = self.preprocess.backward(w=grad)

        return grad.ravel() if grad.is_vector_like else grad

    # TODO: make abstract!
    def _backward(self, w):
        raise NotImplementedError("`_backward` is not implemented for {:}"
                                  "".format(self.__class__.__name__))

    _backward.__doc__ = backward.__doc__  # Same doc for the protected method

    def gradient(self, x, w=None):
        """Compute gradient at x by doing a backward pass.

        Input will be preprocessed first and pre-multiplied by w if provided.

        """
        # Transform data using inner preprocess, if defined
        self._preprocess_data(x, caching=True)
        return self.backward(w)
