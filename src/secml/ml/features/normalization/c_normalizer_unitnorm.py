"""
.. module:: CNormalizerUnitNorm
   :synopsis: Normalize patterns individually to unit norm.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.array import CArray
from secml.ml.features.normalization import CNormalizer


class CNormalizerUnitNorm(CNormalizer):
    """Normalize patterns individually to unit norm.

    Each pattern (i.e. each row of the data matrix) with at least
    one non zero component is rescaled independently of other
    patterns so that its norm (l1 or l2) equals one.

    For the Row normalizer, no training routine is needed, so using
    train_normalize() method is suggested for clarity. Use train() method,
    which does nothing, only to streamline a pipelined environment.

    Parameters
    ----------
    order : {1, 2}, optional
        Order of the norm to normalize each pattern with. Only
        1 ('l1') and 2 ('l2') norm are supported. 2 ('l2') is default.
        For sparse arrays, only 2nd order norm is supported.

    Notes
    -----
    Differently from numpy, we manage flat vectors as 2-Dimensional of
    shape (1, array.size). This means that normalizing a flat vector is
    equivalent to normalize array.atleast_2d(). To obtain a numpy-style
    normalization of flat vectors, transpose array first.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.features.normalization import CNormalizerUnitNorm
    >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

    >>> dense_normalized = CNormalizerUnitNorm().train_normalize(array)
    >>> print dense_normalized
    CArray([[ 0.408248 -0.408248  0.816497]
     [ 1.        0.        0.      ]
     [ 0.        0.707107 -0.707107]])

    >>> print CNormalizerUnitNorm(order=1).train_normalize(array)
    CArray([[ 0.25 -0.25  0.5 ]
     [ 1.    0.    0.  ]
     [ 0.    0.5  -0.5 ]])

    """
    class_type = 'unitnorm'

    def __init__(self, order=2):
        """Class constructor"""
        if order != 1 and order != 2:
            raise ValueError("Norm of order {:} is not supported.".format(order))
        self._order = order
        self._norm = None

    def __clear(self):
        """Reset the object."""
        self._norm = None

    def is_clear(self):
        """Return True if normalizer has not been trained."""
        return self.norm is None

    @property
    def order(self):
        """Returns the order of the norm used for patterns normalization."""
        return self._order

    @property
    def norm(self):
        """Returns the norm of each training array's patterns."""
        return self._norm

    def train(self, x):
        """Train the normalizer. Does reset only.

        For the Row normalizer, no training routine is needed, so using
        train_normalize() method is suggested for clarity. Use train() method,
        which does nothing, only to streamline a pipelined environment.

        Parameters
        ----------
        x : CArray
            Array to be used as training set.
            Each row must correspond to one different pattern.

        Returns
        -------
        CNormalizerRow
            Trained normalizer.

        """
        self.clear()  # Reset trained normalizer

        return self

    def normalize(self, x):
        """Scales array patterns to have unit norm.

        Parameters
        ----------
        x : CArray
            Array to be normalized, 2-Dimensional.

        Returns
        -------
        scaled_array : CArray
            Array with patterns normalized to have unit norm.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.features.normalization import CNormalizerUnitNorm
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]], tosparse=True)

        >>> normalizer = CNormalizerUnitNorm().train(array)
        >>> array_normalized = normalizer.normalize(array)
        >>> print array_normalized  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	0.408248290464
          (0, 1)	-0.408248290464
          (0, 2)	0.816496580928
          (1, 0)	1.0
          (2, 1)	0.707106781187
          (2, 2)	-0.707106781187)
        >>> print array_normalized.todense().norm_2d(order=normalizer.order, axis=1)
        CArray([[ 1.]
         [ 1.]
         [ 1.]])

        """
        x = x.atleast_2d()

        # Computing and storing norm (can be used for revert)
        self._norm = x.norm_2d(order=self.order, axis=1)

        # Makes sure that whenever scale is zero, we handle it correctly
        scale = self._norm.deepcopy()
        scale[scale == 0.0] = 1.0

        if x.issparse:  # Avoid conversion to dense
            x = x.deepcopy().astype(float)
            # Fixes setting floats to int array (result will be float anyway)
            for e_idx, e in enumerate(scale):
                x[e_idx, :] /= e
        else:
            # Normalizing array and removing any 'nan'
            x /= scale  # This creates a copy

        x.nan_to_num()  # Avoid storing nans/inf

        return x

    def revert(self, x):
        """Undo the normalization of data according to training data.

        Parameters
        ----------
        x : CArray
            Array to be reverted, 2-D. Must have been normalized by the same
            calling instance of CNormalizerRow or by a normalizer trained
            with the same data.

        Returns
        -------
        original_array : CArray
            Array with patterns normalized back to original values according
            to training data.

        Notes
        -----
        Due to machine precision errors array returned by revert() is not
        guaranteed to have exactly the same values of original array. To
        solve the problem just use round() function with an arbitrary
        number of decimals.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.features.normalization import CNormalizerUnitNorm
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]], tosparse=True)

        >>> normalizer = CNormalizerUnitNorm().train(array)
        >>> array_normalized = normalizer.normalize(array)
        >>> print normalizer.revert(array_normalized)  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1.0
          (0, 1)	-1.0
          (0, 2)	2.0
          (1, 0)	2.0
          (2, 1)	1.0
          (2, 2)	-1.0)

        """
        x = x.atleast_2d()

        # Training first!
        if self.norm is None:
            raise ValueError("train the normalizer first.")

        if x.shape[0] != self.norm.size:
            raise ValueError("array to revert must have {:} patterns (rows)."
                             "".format(self.norm.size))

        return x * self.norm
