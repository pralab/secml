"""
.. module:: ZScoreScaler
   :synopsis: Scales input array features to have zero mean and unit variance.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.array import CArray
from secml.features.normalization import CNormalizerLinear


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
    >>> from secml.features.normalization import CNormalizerZScore
    >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

    >>> print CNormalizerZScore().train_normalize(array)
    CArray([[ 0.         -1.22474487  1.33630621]
     [ 1.22474487  0.         -0.26726124]
     [-1.22474487  1.22474487 -1.06904497]])

    >>> print CNormalizerZScore(with_std=False).train_normalize(array.tosparse())  # works with sparse arrays too
    CArray([[ 0.         -1.          1.66666667]
     [ 1.          0.         -0.33333333]
     [-1.          1.         -1.33333333]])

    """
    class_type = 'zscore'

    def __init__(self, with_std=True):
        """Class constructor"""
        self._with_std = with_std

        self._data_mean = None
        self._data_std = None
        # Properties of the linear normalizer
        self._w = None
        self._b = None

    def __clear(self):
        """Reset the object."""
        self._data_mean = None
        self._data_std = None
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
        return self._data_mean

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
        return self._data_std

    def train(self, data):
        """Compute the mean and standard deviation to be used for scaling.

        If with_std parameter is set to False, only the mean is calculated.

        Parameters
        ----------
        data : CArray
            Array to be used as training set. Each row must correspond to
            one single patterns, so each column is a different feature.

        Returns
        -------
        trained_scaler : CZScoreScaler
            Scaler trained using input array.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.features.normalization import CNormalizerZScore
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]],tosparse=True)

        >>> normalizer = CNormalizerZScore().train(array)
        >>> print normalizer.mean
        CArray([ 1.          0.          0.33333333])
        >>> print normalizer.std
        CArray([ 0.81649658  0.81649658  1.24721913])

        """
        self.clear()  # Reset trained normalizer
        # Working with CArrays only
        data_array = CArray(data)
        self._data_mean = data_array.mean(axis=0, keepdims=False)
        # Updating linear normalizer parameters
        self._w = CArray.ones(data_array.shape[1])
        self._b = -self._data_mean

        # Updating std deviation parameters only if needed
        if self.with_std is True:
            self._data_std = data_array.std(axis=0, keepdims=False)
            # Updating linear normalizer parameters
            self._w /= self.std
            self._b /= self.std

        return self

    def normalize(self, data):
        """Scales array features to have zero mean and unit variance.

        If with_std parameter is set to False, only the mean is removed.

        Parameters
        ----------
        data : CArray
            Array to be scaled. Must have the same number of features
            (i.e. the number of columns) of training array. If this
            is not the training array, resulting features are not
            guaranteed to have zero mean and/or unit variance

        Returns
        -------
        scaled_array : CArray
            Array with features scaled according to training data.
            Shape of returned array is the same of the original
            array.

        Notes
        -----
        Mean and standard deviation of sparse arrays are stored in dense form.
        As a result of this, if input array is sparse a normalized dense array
        will be always returned.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.features.normalization import CNormalizerZScore
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]], tosparse=True)

        >>> normalizer = CNormalizerZScore().train(array)
        >>> array_normalized = normalizer.normalize(array)
        >>> print array_normalized
        CArray([[ 0.         -1.22474487  1.33630621]
         [ 1.22474487  0.         -0.26726124]
         [-1.22474487  1.22474487 -1.06904497]])
        >>> print array_normalized.mean(axis=0)
        CArray([[ 0.  0.  0.]])
        >>> print array_normalized.std(axis=0)
        CArray([[ 1.  1.  1.]])

        >>> print normalizer.normalize(CArray([-1,5,1]))
        CArray([-2.44948974  6.12372436  0.53452248])
        >>> normalizer.normalize(CArray([-1,5,1]).T)  # We trained on 3 features
        Traceback (most recent call last):
            ...
        ValueError: array to normalize must have 3 features (columns).

        """
        data_scaled = super(CNormalizerZScore, self).normalize(data)

        if self.with_std is not False:
            data_scaled.nan_to_num()  # replacing any nan/inf

        return data_scaled

    def gradient(self, data):
        """Returns the gradient wrt data.

        Gradient of the z-score scaler wrt each row `i` in data is given by:
        .. math::

           \frac{d}{dX_i} = \frac{1}{\sigma}

        if with_std is True. Otherwise, it is simply 1.

        Parameters
        ----------
        data : CArray
            Data array, 2-Dimensional or ravel.

        Returns
        -------
        gradient : CArray
            Gradient of min-max normalizer wrt input data.

        Notes
        -----
        Standard deviation of sparse arrays are stored in dense form.
        As a result of this, if input array is sparse
        a dense array will be always returned.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.features.normalization import CNormalizerZScore
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

        >>> normalizer = CNormalizerZScore().train(array)
        >>> print normalizer.gradient(array)
        CArray([ 1.22474487  1.22474487  0.80178373])

        """
        data_gradient = super(CNormalizerZScore, self).gradient(data)

        if self.with_std is not False:
            data_gradient.nan_to_num()  # replacing any nan/inf

        return data_gradient
