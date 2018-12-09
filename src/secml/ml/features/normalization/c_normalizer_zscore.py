"""
.. module:: CNormalizerZScore
   :synopsis: Scales input array features to have zero mean and unit variance.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.array import CArray
from secml.ml.features.normalization import CNormalizerLinear


class CNormalizerZScore(CNormalizerLinear):
    """Array scaler to zero mean and unit variance.

    This estimator scales and translates each feature individually such
    that the mean of each row is zero and the variance is unit (optional).

    Input data must have one row for each patterns, so features to scale
    are on each array's column.

    The standardization is given by::

        X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0))

    By default, the standard deviation calculated by numpy is the maximum
    likelihood estimate, i.e. the second moment of the set of values about
    their mean. See also :meth:`.CArray.std` for more informations.

    Parameters
    ----------
    with_std : bool, optional
        If True (default), normalizer scales array to have unit variance.

    Notes
    -----
    Mean and standard deviation of sparse arrays are stored in dense form.
    As a result of this, if input array is sparse a normalized dense array
    will be always returned.

    Differently from numpy, we manage flat vectors as 2-Dimensional of
    shape (1, array.size). This means that normalizing a flat vector is
    equivalent to normalize array.atleast_2d(). To obtain a numpy-style
    normalization of flat vectors, transpose array first.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.ml.features.normalization import CNormalizerZScore
    >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

    >>> print CNormalizerZScore().train_normalize(array)
    CArray([[ 0.       -1.224745  1.336306]
     [ 1.224745  0.       -0.267261]
     [-1.224745  1.224745 -1.069045]])

    >>> print CNormalizerZScore(with_std=False).train_normalize(array.tosparse())  # works with sparse arrays too
    CArray([[ 0.       -1.        1.666667]
     [ 1.        0.       -0.333333]
     [-1.        1.       -1.333333]])

    """
    __class_type = 'zscore'

    def __init__(self, with_std=True):
        """Class constructor"""
        self._with_std = with_std

        self._x_mean = None
        self._x_std = None
        # Properties of the linear normalizer
        self._w = None
        self._b = None

    def __clear(self):
        """Reset the object."""
        self._x_mean = None
        self._x_std = None
        # Properties of the linear normalizer
        self._w = None
        self._b = None

    def is_clear(self):
        """Return True if normalizer has not been trained."""
        return self.mean is None and self.std is None and \
            self._w is None and self._b is None

    @property
    def w(self):
        """Returns the slope of the linear normalizer."""
        return self._w

    @property
    def b(self):
        """Returns the bias of the linear normalizer."""
        return self._b

    @property
    def mean(self):
        """Mean of training array per feature.

        Returns
        -------
        train_mean : CArray
            Flat dense array with the mean of each feature
            of the training array. If the scaler has not been
            trained yet, returns None.

        """
        return self._x_mean

    @property
    def with_std(self):
        """Return True if normalizer should transform array to have unit variance."""
        return self._with_std

    @property
    def std(self):
        """Standard deviation of training array per feature.

        Returns
        -------
        train_std : CArray
            Flat dense array with the standard deviation of each
            feature of the training array. If the scaler has not
            been trained yet or with_std is set to False, returns
            None.

        """
        return self._x_std

    def train(self, x):
        """Compute the mean and standard deviation to be used for scaling.

        If with_std parameter is set to False, only the mean is calculated.

        Parameters
        ----------
        x : CArray
            Array to be used as training set. Each row must correspond to
            one single patterns, so each column is a different feature.

        Returns
        -------
        CNormalizerZScore
            Normalizer trained using input array.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.features.normalization import CNormalizerZScore
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]],tosparse=True)

        >>> normalizer = CNormalizerZScore().train(array)
        >>> print normalizer.mean
        CArray([ 1.        0.        0.333333])
        >>> print normalizer.std
        CArray([ 0.816497  0.816497  1.247219])

        """
        self.clear()  # Reset trained normalizer

        self._x_mean = x.mean(axis=0, keepdims=False)

        # Updating linear normalizer parameters
        self._w = CArray.ones(x.shape[1])
        self._b = -self._x_mean

        # Updating std deviation parameters only if needed
        if self.with_std is True:
            self._x_std = x.std(axis=0, keepdims=False)
            # Makes sure that whenever scale is zero, we handle it correctly
            scale = self._x_std.deepcopy()
            scale[scale == 0.0] = 1.0
            # Updating linear normalizer parameters
            self._w /= scale
            self._b /= scale

        return self

    def normalize(self, x):
        """Scales array features to have zero mean and unit variance.

        If with_std parameter is set to False, only the mean is removed.

        Parameters
        ----------
        x : CArray
            Array to be scaled. Must have the same number of features
            (i.e. the number of columns) of training array. If this
            is not the training array, resulting features are not
            guaranteed to have zero mean and/or unit variance

        Returns
        -------
        scaled_array : CArray
            Array with features scaled according to training data.
            Shape of returned array is the same of the original array.

        Notes
        -----
        Mean and standard deviation of sparse arrays are stored in dense form.
        As a result of this, if input array is sparse a normalized dense array
        will be always returned.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.features.normalization import CNormalizerZScore
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]], tosparse=True)

        >>> normalizer = CNormalizerZScore().train(array)
        >>> array_normalized = normalizer.normalize(array)
        >>> print array_normalized
        CArray([[ 0.       -1.224745  1.336306]
         [ 1.224745  0.       -0.267261]
         [-1.224745  1.224745 -1.069045]])
        >>> print array_normalized.mean(axis=0)
        CArray([[ 0.  0.  0.]])
        >>> print array_normalized.std(axis=0)
        CArray([[ 1.  1.  1.]])

        >>> print normalizer.normalize(CArray([-1,5,1]))
        CArray([-2.44949   6.123724  0.534522])
        >>> normalizer.normalize(CArray([-1,5,1]).T)  # We trained on 3 features
        Traceback (most recent call last):
            ...
        ValueError: array to normalize must have 3 features (columns).

        """
        data_scaled = super(CNormalizerZScore, self).normalize(x)

        if self.with_std is not False:
            data_scaled.nan_to_num()  # replacing any nan/inf

        return data_scaled

    def gradient(self, x):
        """Returns the gradient wrt data.

        Gradient of the z-score scaler wrt each row `i` in data is given by:
        .. math::

           \frac{d}{dX_i} = \frac{1}{\sigma}

        if with_std is True. Otherwise, it is simply 1.

        Parameters
        ----------
        x : CArray
            Data array, 2-Dimensional or ravel.

        Returns
        -------
        gradient : CArray
            Gradient of z-score normalizer wrt input data.

        Notes
        -----
        Standard deviation of sparse arrays are stored in dense form.
        As a result of this, if input array is sparse
        a dense array will be always returned.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.features.normalization import CNormalizerZScore
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

        >>> normalizer = CNormalizerZScore().train(array)
        >>> print normalizer.gradient(array)
        CArray([[ 1.224745  0.        0.      ]
         [ 0.        1.224745  0.      ]
         [ 0.        0.        0.801784]])

        """
        data_gradient = super(CNormalizerZScore, self).gradient(x)

        if self.with_std is not False:
            data_gradient.nan_to_num()  # replacing any nan/inf

        return data_gradient
