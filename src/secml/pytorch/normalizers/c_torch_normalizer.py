"""
.. module:: CTorchNormalizer
   :synopsis: Scales input array features to a given range.

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml.array import CArray
from secml.ml.features.normalization import CNormalizer


class CTorchNormalizer(CNormalizer):
    """Normalized features are the CNN deepfeatures

    Parameters
    ----------
    CNN : Neural Network that is going to be used to normalize features
        NB: the CNN network must be already trained

    Attributes
    ----------
    class_type : 'torch-nn'

    """
    __class_type = 'torch-nn'

    def __init__(self, CNN, out_layer=None):
        """Class constructor"""
        self.CNN = CNN
        self.out_layer = out_layer

    def __clear(self):
        """Reset the object."""
        pass

    def is_clear(self):
        """Return True if normalizer has not been trained."""
        return self.CNN.is_clear()

    def train(self, X):
        """Compute the minimum and maximum to be used for scaling.

        Parameters
        ----------
        X : CArray
            CDataset feature values to be used as training set. Each row must correspond to
            one single patterns, so each column is a different feature.

        Returns
        -------
        trained_scaler : CMinMaxScaler
            Scaler trained using input array.

        """
        return self

    def normalize(self, data):
        """Scales array features according to feature_range

        Parameters
        ----------
        data : CArray
            Array to be scaled. Must have the same number of features
            (i.e. the number of columns) of training array. If this is
            not the training array, resulting features can be outside
            feature_range interval.

        Returns
        -------
        data_scaled : CArray
            Array with features scaled according to feature_range and
            training data. Shape of returned array is the same of the
            original array.

        """
        data_scaled = self.CNN.get_layer_output(data, self.out_layer)

        return data_scaled

    def gradient(self, x, w=None):
        """
        Returns the normalizer gradient wrt x.
        The gradient M is a matrix of size m x d, being m the cardinality of
        the normalized feature space, and d that of the input space.

        If the optional argument w is passed (as a m-dimensional vector),
        this function returns w' M.
        If w is set to None (default), M is returned.

        Parameters
        ----------
        x : CArray
            Pattern with respect to which the gradient will be computed.
            Shape (1, n_features) or (n_features, ).

        w : CArray
            Data array, ravel.


        Returns
        -------
        gradient : CArray
            Gradient of the normalizer wrt input data.
            (optionally pre-multiplied by w)

        """
        if not hasattr(self, "_gradient"):
            raise NotImplementedError("gradient is not implemented for {:}"
                                      "".format(self.__class__.__name__))

        if not CArray(x).is_vector_like:
            raise ValueError('Gradient available only wrt a single point!')

        return self._gradient(x, w)

    def _gradient(self, x, w=None):
        """Returns the gradient wrt data.

        clf_output = f( cnn_norm ( norm (x)  )  )

        w = d f( cnn_norm ( norm (x) ) ) / d cnn_norm ( norm (x) ) )

        The derivative is equal to:

        w *  d cnn_norm ( norm (x) ) / d (norm (x) )   *  d norm(x) / d(x)

        Parameters
        ----------
        x : CArray
            Data array, 2-Dimensional or ravel.
        w : CArray (default None)
            classifier gradient

        Returns
        -------
        grad : CArray
            Gradient of min-max normalizer wrt input data.
        """
        # deep featues layer gradient wrt normalized (if normalizer is present) input
        grad = self.CNN.gradient_w_out_x(
            x, w=w.atleast_2d(), layer=self.out_layer)

        # replacing any nan
        grad.nan_to_num()

        return grad

    # (nb)effectively is a shallow copy..
    def __deepcopy__(self, memo, *args, **kwargs):
        return self


        # nb: occhio agli scalari (gli operazioni sugli scalari non vengono tracciate)
        # https://discuss.pytorch.org/t/how-does-one-implement-weight-regularization-l1-or-l2-manually-without-optimum/7951
