"""
.. module:: Function
   :synopsis: Wrapper to manage a function and its gradient

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
.. moduleauthor:: Paolo Russu <paolo.russu@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.core import CCreator
from secml.array import CArray


class CFunction(CCreator):
    """Class that handles generic mathematical functions.

    Either a function or its gradient can be passed in.

    Number of expected space dimensions can be specified if applicable.

    Parameters
    ----------
    fun : any callable or None
        Any python function.
    gradient : any callable or None
        Any python function that returns the gradient of `fun`.
    n_dim : int or None
        Expected space dimensions.

    """
    __super__ = 'CFunction'
    class_type = 'function'

    def __init__(self, fun=None, gradient=None, n_dim=None):

        if fun is None and gradient is None:
            raise ValueError("either `fun` or `gradient` must be passed in.")

        if fun is not None:  # sets function
            self._fun = fun

        if gradient is not None:  # sets gradient of function
            self._gradient = gradient

        # sets expected size of input point `x`
        self._n_dim = n_dim

        CFunction.__clear(self)

    def __clear(self):
        # reset functions evaluation counts
        self._n_fun_eval = 0
        self._n_grad_eval = 0

    @property
    def n_fun_eval(self):
        """Returns the number of function evaluations."""
        return self._n_fun_eval

    @property
    def n_grad_eval(self):
        """Returns the number of gradient evaluations."""
        return self._n_grad_eval

    @property
    def n_dim(self):
        """Returns the expected function's space dimensions."""
        return self._n_dim

    def fun(self, x, *args, **kwargs):
        """Evaluates function at point x.

        Parameters
        ----------
        x : CArray
            Argument of fun.
        args, kwargs : any
            Other optional parameter of the function.

        Returns
        -------
        out_fun : scalar
            Function output, single scalar.

        """
        out_fun = CArray(self._fun(x, *args, **kwargs)).ravel()
        if out_fun.size != 1:
            raise ValueError("function must return a scalar!")
        self._n_fun_eval += 1
        return out_fun[0]

    def fun_ndarray(self, x, *args, **kwargs):
        """Evaluates function at point x (ndarray).

        Parameters
        ----------
        x : ndarray
            Argument of fun as ndarray.
        args, kwargs : any
            Other optional parameter of the function.

        Returns
        -------
        out_fun : scalar
            Function output, single scalar.

        """
        return self.fun(CArray(x), *args, **kwargs)

    def gradient(self, x, *args, **kwargs):
        """Evaluates gradient of function at point x.

        Parameters
        ----------
        x : CArray
            Argument of gradient.
        args, kwargs : any
            Other optional parameter of the function.

        Returns
        -------
        out_grad : CArray
            Array with gradient output.

        """
        out_grad = CArray(self._gradient(x, *args, **kwargs))
        self._n_grad_eval += 1
        return out_grad

    def gradient_ndarray(self, x, *args, **kwargs):
        """Evaluates gradient of function at point x (ndarray).

        Parameters
        ----------
        x : ndarray
            Argument of gradient.
        args, kwargs : any
            Other optional parameter of the function.

        Returns
        -------
        out_grad : ndarray
            Array with gradient output.

        """
        return self.gradient(CArray(x), *args, **kwargs).tondarray()

    def has_fun(self):
        """True if function has been set."""
        return True if hasattr(self, '_fun') else False

    def has_gradient(self):
        """True if gradient has been set."""
        return True if hasattr(self, '_gradient') else False

    def is_equal(self, x, val, tol=1e-6):
        """Evaluates if function value is close to `val` within tol."""
        return True if abs(float(self.fun(x)) - val) <= tol else False
