"""
.. module:: CNormalizerUnitNorm
   :synopsis: Normalize patterns individually to unit norm.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray
from secml.ml.features.normalization import CNormalizer


class CNormalizerUnitNorm(CNormalizer):
    """Normalize patterns individually to unit norm.

    Each pattern (i.e. each row of the data matrix) with at least
    one non zero component is rescaled independently of other
    patterns so that its norm (l1 or l2) equals one.

    For the Row normalizer, no training routine is needed, so using
    fit_normalize() method is suggested for clarity. Use fit() method,
    which does nothing, only to streamline a pipelined environment.

    Parameters
    ----------
    order : {1, 2}, optional
        Order of the norm to normalize each pattern with. Only
        1 ('l1') and 2 ('l2') norm are supported. 2 ('l2') is default.
        For sparse arrays, only 2nd order norm is supported.
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

    >>> dense_normalized = CNormalizerUnitNorm().fit_transform(array)
    >>> print(dense_normalized)
    CArray([[ 0.408248 -0.408248  0.816497]
     [ 1.        0.        0.      ]
     [ 0.        0.707107 -0.707107]])

    >>> print(CNormalizerUnitNorm(order=1).fit_transform(array))
    CArray([[ 0.25 -0.25  0.5 ]
     [ 1.    0.    0.  ]
     [ 0.    0.5  -0.5 ]])

    """
    __class_type = 'unit-norm'

    def __init__(self, order=2, preprocess=None):
        """Class constructor"""
        if order != 1 and order != 2:
            raise ValueError("Norm of order {:} is not supported.".format(order))
        self._order = order

        self._norm = None

        super(CNormalizerUnitNorm, self).__init__(preprocess=preprocess)

    @property
    def order(self):
        """Returns the order of the norm used for patterns normalization."""
        return self._order

    @property
    def norm(self):
        """Returns the norm of each training array's patterns."""
        return self._norm

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

    def _transform(self, x):
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
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]], tosparse=True)

        >>> normalizer = CNormalizerUnitNorm().fit(array)
        >>> array_normalized = normalizer.transform(array)
        >>> print(array_normalized)  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	0.4082482904638631
          (0, 1)	-0.4082482904638631
          (0, 2)	0.8164965809277261
          (1, 0)	1.0
          (2, 1)	0.7071067811865475
          (2, 2)	-0.7071067811865475)
        >>> print(array_normalized.todense().norm_2d(order=normalizer.order, axis=1))
        CArray([[1.]
         [1.]
         [1.]])

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
            x /= scale  # This creates a copy

        return x

    def _revert(self, x):
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

        >>> normalizer = CNormalizerUnitNorm().fit(array)
        >>> array_normalized = normalizer.transform(array)
        >>> print(normalizer.revert(array_normalized))  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1.0
          (0, 1)	-1.0
          (0, 2)	2.0
          (1, 0)	2.0
          (2, 1)	1.0
          (2, 2)	-1.0)

        """
        x = x.atleast_2d()

        if self.norm is None:  # special case of "check_is_fitted"
            raise ValueError(
                "call `.transform` at least one time before using `.revert`.")

        if x.shape[0] != self.norm.size:
            raise ValueError("array to revert must have {:} patterns (rows)."
                             "".format(self.norm.size))

        return x * self.norm
