"""
.. module:: CScalerNorm
   :synopsis: Unit Norm Normalizer.

.. moduleauthor:: Marco Meloni <m.meloni42@studenti.unica.it>

"""
from sklearn.preprocessing import Normalizer

from secml.array import CArray
from secml.ml.scalers import CScalerSkLearn
from secml.core.constants import inf


class CScalerNorm(CScalerSkLearn):
    """CScalerNorm.

    Parameters
    ----------
    norm : ‘l1’, ‘l2’, or ‘max’, optional (‘l2’ by default)
        The norm to use to normalize each non zero sample.
    copy: boolean, optional, default True
        set to False to perform inplace row normalization and avoid a copy.

    Attributes
    ----------
    class_type : 'norm'

    """

    __class_type = 'norm'

    def __init__(self, norm="l2", copy=True, preprocess=None):
        scaler = Normalizer(norm=norm, copy=copy)

        self._order = None
        self.norm = norm

        super(CScalerNorm, self).__init__(
            sklearn_scaler=scaler, preprocess=preprocess)

    def _check_is_fitted(self):
        """This scaler doesn't need fit, so this function doesn't raise any
        exception

        """
        pass

    def _forward(self, x):
        return super(CScalerNorm, self)._forward(x)

    def _backward(self, w=None):
        """Compute the gradient w.r.t. the input cached during the forward
        pass.

        Parameters
        ----------
        w : CArray or None, optional
            If CArray, will be left-multiplied to the gradient
            of the preprocessor.

        Returns
        -------
        gradient : CArray
            Gradient of the normalizer wrt input data.
            it will have dimensionality
            shape (w.shape[0], x.shape[1]) if `w` is passed as input
            (x.shape[1], x.shape[1]) otherwise.

        """
        x = self._cached_x
        if x.shape[0] > 1:
            raise ValueError("Parameter 'x' passed to the forward() method "
                             "needs to be a one dimensional vector "
                             "(passed a {:} dimensional vector)"
                             .format(x.ndim))

        d = self._cached_x.size  # get the number of features
        if w is not None:
            if (w.ndim != 1) or (w.size != d):
                raise ValueError("Parameter 'w' needs to be a one dimensional "
                                 "vector with the same number of elements "
                                 "of parameter 'x' of the forward method "
                                 "(passed a {:} dimensional vector with {:} "
                                 "elements)"
                                 .format(w.ndim, w.size))

        # compute the norm of x: ||x||
        x_norm = self._compute_x_norm(x)
        # compute the gradient of the given norm: d||x||/dx
        grad_norm_x = self._compute_norm_gradient(x, x_norm)

        # this is the derivative of the ratio x/||x||
        grad = CArray.eye(d, d) * x_norm.item() - grad_norm_x.T.dot(x)
        grad /= (x_norm ** 2)

        return grad if w is None else w.dot(grad)

    @property
    def norm(self):
        """Return the norm of each training array's patterns."""
        return self._norm

    @norm.setter
    def norm(self, value):
        """Set the norm that must be used to normalize each row."""
        self._norm = value

        if self._norm == 'l2':
            self._order = 2
        elif self._norm == 'l1':
            self._order = 1
        elif self._norm == "max":
            self._order = inf
        else:
            raise ValueError("unknown norm")

    def _compute_x_norm(self, x):
        """Compute the norm of x: ||x||."""
        x_norm = x.norm_2d(axis=1, keepdims=True, order=self._order)
        x_norm[x_norm == 0] = 1  # to avoid nan values
        return x_norm

    def _compute_norm_gradient(self, x, x_norm):
        """Compute the gradient of the chosen norm on x.

        Parameters
        ----------
        x : CArray
            The input sample.
        x_norm : CArray
            Array containing its pre-computed norm ||x||.

        Returns
        -------
        CArray
            The derivative d||x||/dx of the chosen norm.

        """
        d = x.size  # number of features
        if self.norm == "l2":
            grad_norm_x = x / x_norm
        elif self.norm == "l1":
            sign = x.sign()
            grad_norm_x = sign
        elif self.norm == 'max':
            grad_norm_x = CArray.zeros(d, sparse=x.issparse)
            abs_x = x.abs()  # take absolute values of x...
            max_abs_x = abs_x.max()  # ... and the maximum absolute value
            max_abs_x -= 1e-8  # add small tolerance
            max_idx = abs_x >= max_abs_x  # find idx of maximum values
            grad_norm_x[max_idx] = x[max_idx].sign()
        else:
            raise ValueError("Unsupported norm.")
        # return the gradient of ||x||
        return grad_norm_x
