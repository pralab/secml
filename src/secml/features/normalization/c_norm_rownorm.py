"""
.. module:: CRowNormalizer
   :synopsis: Normalize patterns individually to unit norm.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.array import CArray
from secml.features.normalization import CNormalizer


class CNormalizerRow(CNormalizer):
    """Normalize patterns individually to unit norm.

    Each pattern (i.e. each row of the data matrix) with at least
    one non zero component is rescaled independently of other
    patterns so that its norm (l1 or l2) equals one.

    For the Row normalizer, no training routine is needed, so using
    train_normalize() method is suggested for clarity. Use train() method,
    which does nothing, only to streamline a pipelined environment.

    Parameters
    ----------
    ord : {1, 2}, optional
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
    >>> from secml.features.normalization import CNormalizerRow
    >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

    >>> dense_normalized = CNormalizerRow().train_normalize(array)
    >>> print dense_normalized
    CArray([[ 0.40824829 -0.40824829  0.81649658]
     [ 1.          0.          0.        ]
     [ 0.          0.70710678 -0.70710678]])

    >>> print CNormalizerRow(ord=1).train_normalize(array)
    CArray([[ 0.25 -0.25  0.5 ]
     [ 1.    0.    0.  ]
     [ 0.    0.5  -0.5 ]])

    """
    class_type = 'rownorm'

    def __init__(self, ord=2):
        """Class constructor"""
        if ord != 1 and ord != 2:
            raise ValueError("Norm of order {:} is not supported.".format(ord))
        self._ord = ord
        self._norm = None

    def __clear(self):
        """Reset the object."""
        self._norm = None

    def is_clear(self):
        """Return True if normalizer has not been trained."""
        return self.norm is None

    @property
    def ord(self):
        """Returns the order of the norm used for patterns normalization."""
        return self._ord

    @property
    def norm(self):
        """Returns the norm of each training array's patterns."""
        return self._norm

    def train(self, data):
        """Train the normalizer. Does reset only.

        For the Row normalizer, no training routine is needed, so using
        train_normalize() method is suggested for clarity. Use train() method,
        which does nothing, only to streamline a pipelined environment.

        Parameters
        ----------
        data : CArray
            Array to be used as training set.
            Each row must correspond to one different pattern.

        Returns
        -------
        trained_normalizer : CRowNormalizer
            Reset normalizer.

        """
        self.clear()  # Reset trained normalizer

        return self

    def normalize(self, data):
        """Scales array patterns to have unit norm.

        Parameters
        ----------
        data : CArray
            Array to be normalized.

        Returns
        -------
        scaled_array : CArray
            Array with patterns normalized to have unit norm.
            Shape of returned array is the same of the original array.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.features.normalization import CNormalizerRow
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]], tosparse=True)

        >>> normalizer = CNormalizerRow().train(array)
        >>> array_normalized = normalizer.normalize(array)
        >>> print array_normalized  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	0.408248290464
          (0, 1)	-0.408248290464
          (0, 2)	0.816496580928
          (1, 0)	1.0
          (2, 1)	0.707106781187
          (2, 2)	-0.707106781187)
        >>> print array_normalized.todense().norm_2d(ord=normalizer.ord, axis=1)
        CArray([[ 1.]
         [ 1.]
         [ 1.]])

        """
        data_array = CArray(data)  # working on CArrays

        # Computing and storing norm (can be used for revert)
        self._norm = CArray(data_array.norm_2d(ord=self.ord, axis=1))

        if data_array.issparse:  # Avoid conversion to dense
            data_array = data_array.deepcopy().astype(float)
            # Fixes setting floats to int array (result will be float anyway)
            for e_idx, e in enumerate(self._norm):
                res = CArray(data_array[e_idx, :]) / e
                data_array[e_idx, :] = res
        else:
            # Normalizing array and removing any 'nan'
            data_array /= self.norm  # This creates a copy

        data_array.nan_to_num()  # Avoid storing nans/inf

        return data_array

    def revert(self, data):
        """Undo the normalization of data according to training data.

        Parameters
        ----------
        data : CArray
            Array to be reverted. Must have been normalized by the same
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
        >>> from secml.features.normalization import CNormalizerRow
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]], tosparse=True)

        >>> normalizer = CNormalizerRow().train(array)
        >>> array_normalized = normalizer.normalize(array)
        >>> print normalizer.revert(array_normalized)  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1.0
          (0, 1)	-1.0
          (0, 2)	2.0
          (1, 0)	2.0
          (2, 1)	1.0
          (2, 2)	-1.0)

        """
        data_array = CArray(data)  # working on CArrays
        # Training first!
        if self.norm is None:
            raise ValueError("train the normalizer first.")
        if data_array.atleast_2d().shape[0] != self.norm.size:
            raise ValueError("array to revert must have {:} patterns (rows)."
                             "".format(self.norm.size))

        return data_array * self.norm
