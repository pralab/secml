"""
.. module:: CNormalizerMinMax
   :synopsis: Scales input array features to a given range.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray
from secml.ml.features.normalization import CNormalizerLinear


class CNormalizerMinMax(CNormalizerLinear):
    """Standardizes array by scaling each feature to a given range.

    This estimator scales and translates each feature
    individually such that it is in the given range on
    the training array, i.e. between zero and one.

    Input data must have one row for each patterns,
    so features to scale are on each array's column.

    The standardization is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    Parameters
    ----------
    feature_range : tuple of scalars or None, optional
        Desired range of transformed data, tuple of 2 scalars where
        `feature_range[0]` is the minimum and `feature_range[1]` is
        the maximum value. If feature_range is None, features will be
        scaled using (0., 1.) range.
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'min-max'

    Notes
    -----
    Differently from numpy, we manage flat vectors as 2-Dimensional of
    shape (1, array.size). This means that normalizing a flat vector is
    equivalent to transform array.atleast_2d(). To obtain a numpy-style
    normalization of flat vectors, transpose array first.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.features.normalization import CNormalizerMinMax
    >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

    >>> print(CNormalizerMinMax().fit_transform(array))
    CArray([[0.5      0.       1.      ]
     [1.       0.5      0.333333]
     [0.       1.       0.      ]])

    >>> print(CNormalizerMinMax(feature_range=(-1,1)).fit_transform(array))
    CArray([[ 0.       -1.        1.      ]
     [ 1.        0.       -0.333333]
     [-1.        1.       -1.      ]])

    """
    __class_type = 'min-max'

    def __init__(self, feature_range=None, preprocess=None):

        self._feature_range = None
        # setting desired feature range... the property will check for correct type
        self.feature_range = (0., 1.) if feature_range is None else feature_range

        self._min = None
        self._max = None
        # Properties of the linear normalizer
        # we split them to easily manage feature_range
        self._m = None
        self._q = None

        super(CNormalizerMinMax, self).__init__(preprocess=preprocess)

    @property
    def w(self):
        """Returns the slope of the linear normalizer."""
        n = self.feature_range[1] - self.feature_range[0]
        return n * self._m

    @property
    def b(self):
        """Returns the bias of the linear normalizer."""
        n = self.feature_range[1] - self.feature_range[0]
        v = self.feature_range[0]
        return n * self._q + v

    @property
    def m(self):
        """Returns the slope of the linear normalizer.
        (excluding the feature range)."""
        return self._m

    @property
    def q(self):
        """Returns the bias of the linear normalizer
        (excluding the feature range)."""
        return self._q

    @property
    def min(self):
        """Minimum of training array per feature.

        Returns
        -------
        train_min : CArray
            Flat dense array with the minimum of each feature
            of the training array. If the scaler has not been
            trained yet, returns None.

        """
        return self._min

    @property
    def max(self):
        """Maximum of training array per feature.

        Returns
        -------
        train_max : CArray
            Flat dense array with the maximum of each feature
            of the training array. If the scaler has not been
            trained yet, returns None.

        """
        return self._max

    @property
    def feature_range(self):
        """Desired range of transformed data."""
        return self._feature_range

    @feature_range.setter
    def feature_range(self, feature_range):
        """Set the desired range of transformed data.

        Parameters
        ----------
        feature_range : tuple of scalars
            Desired range of transformed data, tuple of 2 scalars where
            feature_range[0] is the minimum and feature_range[1] is the
            maximum value.

        """
        if not isinstance(feature_range, tuple) and len(feature_range) == 2:
            raise TypeError("feature range must be a tuple of 2 scalars.")
        else:
            self._feature_range = feature_range

    def _fit(self, x, y=None):
        """Compute the minimum and maximum to be used for scaling.

        Parameters
        ----------
        x : CArray
            Array to be used as training set. Each row must correspond to
            one single patterns, so each column is a different feature.
        y : CArray or None, optional
            Flat array with the label of each pattern.
            Can be None if not required by the preprocessing algorithm.

        Returns
        -------
        CNormalizerMinMax
            Instance of the trained normalizer.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.features.normalization import CNormalizerMinMax
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

        >>> normalizer = CNormalizerMinMax().fit(array)
        >>> normalizer.feature_range
        (0.0, 1.0)
        >>> print(normalizer.min)
        CArray([ 0. -1. -1.])
        >>> print(normalizer.max)
        CArray([2. 1. 2.])

        """
        self._min = x.min(axis=0, keepdims=False)
        self._max = x.max(axis=0, keepdims=False)

        # Setting the linear normalization properties
        # y = m * x + q
        r = CArray(self.max - self.min)
        self._m = CArray.ones(r.size)
        self._m[r != 0] = 1.0 / r[r != 0]  # avoids division by zero
        self._q = -self.min * self._m
        # z = n * y + v  ->  Y = n * m * x + (n * q + v)

        return self
