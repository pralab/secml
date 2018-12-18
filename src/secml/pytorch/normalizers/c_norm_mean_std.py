from secml.array import CArray
from secml.core.type_utils import is_scalar
from secml.ml.features.normalization import CNormalizerLinear


# TODO: COMPLETE DOCSTRINGS. ADD UNITTESTS
class CNormalizerMeanSTD(CNormalizerLinear):
    """Normalize with given mean and standard deviation.

    If mean/std are lists of multiple values, input is expected to be
    uniformly splittable in a number of channel equal to the number of
    input values.

    Result will be: (input[channel] - mean[channel]) / std[channel]

    Parameters
    ----------
    mean : scalar or tuple
    std : scalar or tuple

    Attributes
    ----------
    class_type : 'mean-std'

    """
    __class_type = 'mean-std'

    def __init__(self, mean, std):

        self._mean = (mean, ) if is_scalar(mean) else tuple(mean)
        self._std = (std, ) if is_scalar(std) else tuple(std)

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

    def __is_clear(self):
        """Returns True if object is clear."""
        return self._w is None and self._b is None and \
            self._data_mean is None and self._data_std is None

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
        """Mean."""
        return self._mean

    @property
    def std(self):
        """Standard deviation."""
        return self._std

    def train(self, data):
        """Compute the mean and standard deviation to be used for scaling.

        Parameters
        ----------
        data : CArray
            Array to be used as training set. Each row must correspond to
            one single patterns, so each column is a different feature.

        Returns
        -------
        CNormalizerMeanSTD
            Scaler trained using input array.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.pytorch.normalizers import CNormalizerMeanSTD
        >>> array = CArray([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]],tosparse=True)

        >>> normalizer = CNormalizerMeanSTD(0.5, 0.2).train(array)
        >>> print normalizer._data_mean
        CArray([ 0.5  0.5  0.5])
        >>> print normalizer._data_std
        CArray([ 0.2  0.2  0.2])

        >>> print normalizer.normalize(array)
        CArray([[ 2.5 -7.5  7.5]
         [ 7.5 -2.5 -2.5]
         [-2.5  2.5 -7.5]])

        >>> normalizer = CNormalizerMeanSTD((0.5, 0.5, 0.2), (0.2, 0.1, 0.1)).train(array)

        >>> print normalizer.normalize(array)
        CArray([[  2.5 -15.   18. ]
         [  7.5  -5.   -2. ]
         [ -2.5   5.  -12. ]])

        """
        self.clear()  # Reset trained normalizer
        # Working with CArrays only
        data_array = CArray(data).atleast_2d()
        n_feats = data_array.shape[1]

        n_channels = len(self.mean)
        if not n_feats % n_channels == 0:
            raise ValueError("input number of features must be "
                             "divisible by {:}".format(n_channels))
        mean_list = []
        std_list = []
        for i in xrange(n_channels):
            mean_list.append(
                CArray.ones(shape=(n_feats / n_channels, )) * self.mean[i])
            std_list.append(
                CArray.ones(shape=(n_feats / n_channels, )) * self.std[i])
        self._data_mean = CArray.from_iterables(mean_list)
        self._data_std = CArray.from_iterables(std_list)

        # Updating linear normalizer parameters
        self._w = CArray.ones(n_feats)
        self._b = -self._data_mean

        # Updating linear normalizer parameters
        self._w /= self._data_std
        self._b /= self._data_std

        return self

    def normalize(self, data):
        """Linearly scales array features.

        Parameters
        ----------
        data : CArray
            Array to be scaled. Must have the same number of features
            (i.e. the number of columns) of training array.

        Returns
        -------
        scaled_array : CArray
            Array with features linearly scaled.
            Shape of returned array is the same of the original array.

        """
        if self.is_clear():
            # This normalize has no "real" train so we can do it right now
            self.train(data)
        return super(CNormalizerMeanSTD, self).normalize(data)
