

"""
.. module:: NormalizerMinMax
   :synopsis: Scales input array features to a given range.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from prlib.array import CArray
from prlib.features.normalization import CNormalizerLinear
from prlib.core.constants import inf


# TODO: ADD SPARSE ARRAYS SUPPORT
class CNormalizerMinMax(CNormalizerLinear):
    """Standardizes array by scaling each feature to a given range.

    This normalizer scales and shifts each feature
    individually such that it belong in the given range on
    the training array, i.e. between zero and one.

    Input data is supposed to have one row for each patterns.
    Flat array are considered as row array (One single sample with a number
    of features equal to the array size)

    The standardization is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    .. warning::

        Currently this normalizer works correctly with dense arrays only.

    Parameters
    ----------
    feature_range : tuple of scalars or None, optional
        Desired range of transformed data, tuple of 2 scalars where
        `feature_range[0]` is the minimum and `feature_range[1]` is
        the maximum value. If feature_range is None, features will be
        scaled using (0., 1.) range.

    Notes
    -----
    Differently from numpy, we manage flat vectors as 2-Dimensional of
    shape (1, array.size). This means that normalizing a flat vector is
    equivalent to normalize array.atleast_2d(). To obtain a numpy-style
    normalization of flat vectors, transpose array first.

    Examples
    --------
    >>> from prlib.array import CArray
    >>> from prlib.features.normalization import CNormalizerMinMax
    >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

    >>> print CNormalizerMinMax().train_normalize(array)
    CArray([[ 0.5         0.          1.        ]
     [ 1.          0.5         0.33333333]
     [ 0.          1.          0.        ]])

    >>> print CNormalizerMinMax(feature_range=(-1,1)).train_normalize(array)
    CArray([[ 0.         -1.          1.        ]
     [ 1.          0.         -0.33333333]
     [-1.          1.         -1.        ]])

    """
    class_type = 'minmax'

    def __init__(self, feature_range=None):
        """Class constructor"""
        self._data_min = None
        self._data_max = None
        # Properties of the linear normalizer
        # we split them to easily manage feature_range
        self._m = None
        self._q = None
        # The following two shouldn't be reset:
        # feature range does not depends on training
        self._n = None
        self._v = None

        self._feature_range = None
        # setting desired feature range... the property will check for correct type
        self.feature_range = (0., 1.) if feature_range is None else feature_range

    def __clear(self):
        """Reset the object."""
        self._data_min = None
        self._data_max = None
        # Properties of the linear normalizer
        # we split them to easily manage feature_range
        self._m = None
        self._q = None

    def is_clear(self):
        """Return True if normalizer has not been trained."""
        return self.min is None and self.max is None and \
            self._m is None and self._q is None

    @property
    def w(self):
        """Returns the slope of the linear normalizer."""
        return self._n * self._m

    @property
    def b(self):
        """Returns the bias of the linear normalizer."""
        return self._n * self._q + self._v

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
        return self._data_min

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
        return self._data_max

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
            # Resetting parameters associated with feature range
            self._n = self.feature_range[1] - self.feature_range[0]
            self._v = self.feature_range[0]

    def train(self, data):
        """Compute the minimum and maximum to be used for scaling.

        Parameters
        ----------
        data : CArray
            Array to be used as training set. Each row must correspond to
            one single patterns, so each column is a different feature.
            Flat array are considered as row array (One single sample with
            feature number equal to the array size).

        Returns
        -------
        trained_scaler : CMinMaxScaler
            Scaler trained using input array.

        Examples
        --------
        >>> from prlib.array import CArray
        >>> from prlib.features.normalization import CNormalizerMinMax
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

        >>> normalizer = CNormalizerMinMax().train(array)
        >>> normalizer.feature_range
        (0.0, 1.0)
        >>> print normalizer.min
        CArray([ 0. -1. -1.])
        >>> print normalizer.max
        CArray([ 2.  1.  2.])

        """
        self.clear()  # Reset trained normalizer
        # Working with CArrays only
        data_array = CArray(data)
        if data_array.issparse:
            raise NotImplementedError(
                "normalization of sparse arrays is not yet supported!")

        self._data_min = data_array.min(axis=0, keepdims=False)
        self._data_max = data_array.max(axis=0, keepdims=False)
        # Setting the linear normalization properties
        # y = m * x + q
        r = CArray(self.max - self.min)
        self._m = CArray.zeros(r.size)
        self._m[r != 0] = 1.0 / r[r != 0]  # avoids division by zero
        self._q = -self.min * self._m
        # z = n * y + v  ->  Y = n * m * x + (n * q + v)

        return self

    def normalize(self, data):
        """Scales array features according to feature_range.

        Parameters
        ----------
        data : CArray
            Array to be scaled. Must have the same number of features
            (i.e. the number of columns) of training array. If this is
            not the training array, resulting features can be outside
            feature_range interval.
            Flat array are considered as row array (One single sample with
            feature number equal to the array size).

        Returns
        -------
        scaled_array : CArray
            Array with features scaled according to feature_range and
            training data. Shape of returned array is the same of the
            original array.

        Examples
        --------
        >>> from prlib.array import CArray
        >>> from prlib.features.normalization import CNormalizerMinMax
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

        >>> normalizer = CNormalizerMinMax().train(array)
        >>> print normalizer.normalize(array)
        CArray([[ 0.5         0.          1.        ]
         [ 1.          0.5         0.33333333]
         [ 0.          1.          0.        ]])

        >>> print normalizer.normalize(CArray([-1,5,1]))
        CArray([-0.5         3.          0.66666667])
        >>> normalizer.normalize(CArray([-1,5,1]).T)  # We trained on 3 features
        Traceback (most recent call last):
            ...
        ValueError: array to normalize must have 3 features (columns).

        """
        data_scaled = super(CNormalizerMinMax, self).normalize(data)

        # Setting values outside feature_range to the bound
        data_scaled[data_scaled < self.feature_range[0]] = self.feature_range[0]
        data_scaled[data_scaled > self.feature_range[1]] = self.feature_range[1]
        # replacing any nan
        data_scaled.nan_to_num()

        return data_scaled

    def gradient(self, data):
        """Returns the gradient wrt data.

        Gradient of the min-max scaler wrt each row `i` in data is given by:
        .. math::

           \frac{d}{dX_i} = \frac{feat_range[1] - feat_range[0]}{max - min}

        Parameters
        ----------
        data : CArray
            Data array, 2-Dimensional or ravel.

        Returns
        -------
        gradient : CArray
            Gradient of min-max normalizer wrt input data.

        Examples
        --------
        >>> from prlib.array import CArray
        >>> from prlib.features.normalization import CNormalizerMinMax
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

        >>> normalizer = CNormalizerMinMax().train(array)
        >>> print normalizer.gradient(array)
        CArray([ 0.5         0.5         0.33333333])

        """
        data_gradient = super(CNormalizerMinMax, self).gradient(data)

        # Replacing any inf with proper values
        data_gradient[data_gradient == -inf] = self.feature_range[0]
        data_gradient[data_gradient == inf] = self.feature_range[1]
        # replacing any nan
        data_gradient.nan_to_num()

        return data_gradient

