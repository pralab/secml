"""
.. module:: CFunction
   :synopsis: Wrapper to manage a function and its gradient

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
.. moduleauthor:: Paolo Russu <paolo.russu@diee.unica.it>

"""
from secml.core import CCreator
from secml.core.type_utils import is_scalar
from secml.array import CArray

from scipy import optimize as sc_opt
from secml.core.constants import eps


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

    Attributes
    ----------
    class_type : 'standard'

    """
    __super__ = 'CFunction'
    __class_type = 'generic'

    def __init__(self, fun=None, gradient=None, n_dim=None):

        if fun is None and gradient is None:
            raise ValueError("either `fun` or `gradient` must be passed in.")

        if fun is not None:  # sets function
            self._fun = fun

        if gradient is not None:  # sets gradient of function
            self._gradient = gradient

        # sets expected size of input point `x`
        self._n_dim = n_dim

        self._n_fun_eval = 0
        self._n_grad_eval = 0

    def __clear(self):
        """Reset the object."""
        self._n_fun_eval = 0
        self._n_grad_eval = 0

    def __is_clear(self):
        """Returns True if object is clear."""
        return self._n_fun_eval + self._n_grad_eval == 0

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
        out_fun = self._fun(x, *args, **kwargs)

        # Function can return a CArray of size 1 or a scalar
        if isinstance(out_fun, CArray):
            if out_fun.size != 1:
                raise ValueError("`_fun` must return a CArray of size 1!")
            out_fun = out_fun.item()
        elif is_scalar(out_fun):
            pass  # Returned a scalar, OK
        else:
            raise TypeError(
                "`_fun` must return a scalar or a CArray of size 1. "
                "Returned a {:}".format(type(out_fun)))

        self._n_fun_eval += 1
        return out_fun

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
        out_grad = self._gradient(x, *args, **kwargs)
        if not isinstance(out_grad, CArray):
            raise TypeError("`_gradient` must return a CArray!")
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

    def approx_fprime(self, x, epsilon, *args):
        """Finite-difference approximation of the gradient of a scalar function.

        Wrapper for scipy function :func:`scipy.optimize.approx_fprime`.

        Parameters
        ----------
        x : CArray
            The flat dense vector with the point at which to determine
            the gradient of `fun`.
        epsilon : scalar or CArray
            Increment of `x` to use for determining the function gradient.
            If a scalar, uses the same finite difference delta for all partial
            derivatives.
            If an array, should contain one value per element of `x`.
        *args : args, optional
            Any other arguments that are to be passed to `fun`.

        Returns
        -------
        grad : CArray
            The gradient of `fun` at `x`.

        See Also
        --------
        `.check_grad` : Check correctness of function gradient
            against `approx_fprime`.

        Notes
        -----
        The function gradient is determined by the forward finite difference
        formula::

                     fun(xk[i] + epsilon[i]) - f(xk[i])
           fun'[i] = -----------------------------------
                                epsilon[i]

        The main use of `approx_fprime` is to determine numerically
        the Jacobian of a function.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.optim.function import CFunction
        >>> from secml.core.constants import eps

        >>> def func(x, c0, c1):
        ...     "Coordinate vector `x` should be an array of size two."
        ...     return c0 * x[0]**2 + c1*x[1]**2

        >>> c0, c1 = (1, 200)
        >>> CFunction(func).approx_fprime(CArray.ones(2), [eps, (200 ** 0.5) * eps], c0, c1)
        CArray(2,)(dense: [   2.        400.000042])

        """
        if x.issparse is True or x.is_vector_like is False:
            raise ValueError("x0 must be a dense flat array")

        # double casting to always have a CArray
        xk_ndarray = CArray(x).ravel().tondarray()

        epsilon = epsilon.tondarray() if \
            isinstance(epsilon, CArray) else epsilon

        return CArray(
            sc_opt.approx_fprime(xk_ndarray, self.fun_ndarray, epsilon, *args))

    def check_grad(self, x, *args, **epsilon):
        """Check the correctness of a gradient function by comparing
         it against a (forward) finite-difference approximation of
         the gradient.

        Parameters
        ----------
        x : CArray
            Flat dense pattern to check function gradient against
            forward difference approximation of function gradient.
        epsilon : scalar or CArray
            Increment to `x` to use for determining the function gradient.
            If a scalar, uses the same finite difference delta for all partial
            derivatives.  If an array, should contain one value per element of
            `x`.
        *args : *args, optional
            Extra arguments passed to `fun` and `fprime`.

        Returns
        -------
        err : float
            The square root of the sum of squares (i.e. the l2-norm) of the
            difference between ``fprime(x, *args)`` and the finite difference
            approximation of `fprime` at the points `x`.

        Notes
        -----
        `epsilon` is the only keyword argument accepted by the function. Any
        other optional argument for `fun` and `fprime` should be passed as
        non-keyword.

        See Also
        --------
        `.approx_fprime` : Finite-difference approximation of the gradient of a scalar function.

        Examples
        --------
        >>> from secml.optim.function import CFunction
        >>> from secml.array import CArray

        >>> def func(x):
        ...     return x[0].item()**2 - 0.5 * x[1].item()**3
        >>> def grad(x):
        ...     return CArray([2 * x[0].item(), -1.5 * x[1].item()**2])

        >>> fun = CFunction(func, grad)
        >>> fun.check_grad(CArray([1.5, -1.5]))
        2.9802322387695312e-08

        """
        if x.issparse is True or x.is_vector_like is False:
            raise ValueError("x0 must be a dense flat array")

        # We now extract 'epsilon' if passed by the user
        if 'epsilon' in epsilon:
            epsilon = epsilon.pop('epsilon', eps)
        else:
            epsilon = eps

        # real value of the gradient on x
        grad = self.gradient(x, *args)
        # value of the approximated gradient on x
        approx = self.approx_fprime(x, epsilon, *args)

        return (grad - approx).norm()
