"""
.. module:: CNormalizerUnitNorm
   :synopsis: Normalize patterns individually to unit norm.

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from secml.array import CArray
from secml.ml.features.normalization import CNormalizer
from secml.core.constants import inf


class CNormalizerUnitNorm(CNormalizer):
    """Normalize patterns individually to unit norm.

    Each pattern (i.e. each row of the data matrix) with at least
    one non zero component is rescaled independently of other
    patterns so that its norm (l1 or l2 or max) equals one.

    For the Row normalizer, no training routine is needed, so using
    fit_normalize() method is suggested for clarity. Use fit() method,
    which does nothing, only to streamline a pipelined environment.

    Parameters
    ----------
    norm : {'l1', 'l2', 'max'}, optional
        Order of the norm to normalize each pattern with.'l2' is the default.
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'unit-norm'

    Notes
    -----
    Differently from numpy, we manage flat vectors as 2-Dimensional of
    shape (1, array.size). This means that normalizing a flat vector is
    equivalent to transform array.atleast_2d(). To obtain a numpy-style
    normalization of flat vectors, transpose array first.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.features.normalization import CNormalizerUnitNorm
    >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

    >>> dense_normalized = CNormalizerUnitNorm(norm="l2").fit_transform(array)
    >>> print(dense_normalized)
    CArray([[ 0.408248 -0.408248  0.816497]
     [ 1.        0.        0.      ]
     [ 0.        0.707107 -0.707107]])

    >>> dense_normalized =(CNormalizerUnitNorm(norm="l1").fit_transform(array))
    >>> print(dense_normalized)
    CArray([[ 0.25 -0.25  0.5 ]
     [ 1.    0.    0.  ]
     [ 0.    0.5  -0.5 ]])

    >>> print(array / array.norm_2d(order=1, axis=1, keepdims=True))
    CArray([[ 0.25 -0.25  0.5 ]
     [ 1.    0.    0.  ]
     [ 0.    0.5  -0.5 ]])

    """
    __class_type = 'unit-norm'

    def __init__(self, norm="l2", preprocess=None):
        """Class constructor"""
        self._order = None
        self.norm = norm
        super(CNormalizerUnitNorm, self).__init__(preprocess=preprocess)

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

    def _check_is_fitted(self):
        """Check if the preprocessor is trained (fitted).

        Raises
        ------
        NotFittedError
            If the preprocessor is not fitted.

        """
        pass  # This preprocessor does not require training

    def _fit(self, x, y=None):
        """Fit the normalizer.

        For the Row normalizer, no training routine is needed, so using
        fit_transform() method is suggested for clarity. Use fit() method,
        which does nothing, only to streamline a pipelined environment.

        Parameters
        ----------
        x : CArray
            Array to be used as training set.
            Each row must correspond to one different pattern.
        y : CArray or None, optional
            Flat array with the label of each pattern.
            Can be None if not required by the preprocessing algorithm.

        Returns
        -------
        CNormalizerRow
            Instance of the trained normalizer.

        """
        return self

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

    def _forward(self, x):
        """Transform array patterns to have unit norm.

        Parameters
        ----------
        x : CArray
            Array to be normalized, 2-Dimensional.

        Returns
        -------
        CArray
            Array with patterns normalized to have unit norm.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.features.normalization import CNormalizerUnitNorm
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])
        >>> array = array.tosparse()

        >>> normalizer = CNormalizerUnitNorm().fit(array)
        >>> array_normalized = normalizer.transform(array)
        >>> print(array_normalized)  # doctest: +NORMALIZE_WHITESPACE
        CArray([[ 0.408248 -0.408248  0.816497]
         [ 1.        0.        0.      ]
         [ 0.        0.707107 -0.707107]])
        >>> print(array_normalized.todense().norm_2d(order=2, axis=1))
        CArray([[1.]
         [1.]
         [1.]])

        """
        x_norm = self._compute_x_norm(x)
        # fixme: if you do x/x_norm with x sparse, result is dense
        #  this needs patching in CArray
        return 1.0 / x_norm * x

    def _backward(self, w=None):
        """
        Compute the gradient w.r.t. the input cached during the forward pass.

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
        d = self._cached_x.size  # get the number of features

        # compute the norm of x: ||x||
        x_norm = self._compute_x_norm(x)
        # compute the gradient of the given norm: d||x||/dx
        grad_norm_x = self._compute_norm_gradient(x, x_norm)

        # this is the derivative of the ratio x/||x||
        grad = CArray.eye(d, d) * x_norm.item() - grad_norm_x.T.dot(x)
        grad /= (x_norm ** 2)

        return grad if w is None else w.dot(grad)
