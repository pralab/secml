"""
.. module:: LinearNormalizers
   :synopsis: Common interface for linear normalizers.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from abc import abstractproperty
from secml.array import CArray
from secml.ml.features.normalization import CNormalizer


# TODO: ADD SPARSE ARRAYS SUPPORT
class CNormalizerLinear(CNormalizer):
    """Standardizes array by scaling linearly each feature.

    Input data must have one row for each patterns,
    so features to scale are on each array's column.

    The standardization is given by::

        X_scaled = m * X(axis=0) + q

    where m, q are specific constants for each normalization.

    .. warning::

        Currently only few linear normalizers work with sparse arrays.

    Notes
    -----
    Only arrays of dense form are supported.

    Differently from numpy, we manage flat vectors as 2-Dimensional of
    shape (1, array.size). This means that normalizing a flat vector is
    equivalent to normalize array.atleast_2d(). To obtain a numpy-style
    normalization of flat vectors, transpose array first.

    """
    def is_linear(self):
        """Returns True for linear normalizers."""
        return True

    @abstractproperty
    def w(self):
        """Returns the step of the linear normalizer."""
        # w must be a CArray
        raise NotImplementedError("Linear normalizer should define the slope.")

    @abstractproperty
    def b(self):
        """Returns the bias of the linear normalizer."""
        # b must be a CArray
        raise NotImplementedError("Linear normalizer should define the bias.")

    def normalize(self, x):
        """Linearly scales array features.

        Parameters
        ----------
        x : CArray
            Array to be scaled. Must have the same number of features
            (i.e. the number of columns) of training array.

        Returns
        -------
        scaled_array : CArray
            Array with features linearly scaled.
            Shape of returned array is the same of the original array.

        """
        # Training first!
        if self.is_clear():
            raise ValueError("train the normalizer first.")

        if x.atleast_2d().shape[1] != self.w.size:
            raise ValueError("array to normalize must have {:} "
                             "features (columns).".format(self.w.size))

        return self.w * x + self.b

    def revert(self, x):
        """Undo the linear normalization of input data.

        Parameters
        ----------
        x : CArray
            Array to be reverted. Must have been normalized by the same
            calling instance of the CNormalizerLinear.

        Returns
        -------
        original_array : CArray
            Array with features scaled back to original values.

        """
        # Training first!
        if self.is_clear():
            raise ValueError("train the normalizer first.")
        if x.atleast_2d().shape[1] != self.w.size:
            raise ValueError("array to revert must have {:} "
                             "features (columns).".format(self.w.size))

        v = (x - self.b) / self.w

        # set nan/inf to zero
        zeros_feats = self.w.find(self.w == 0)
        if len(zeros_feats) > 0:
            if v.ndim == 1:
                v[zeros_feats] = 0
            else:
                v[:, zeros_feats] = 0

        return v

    def gradient(self, x):
        """Returns the gradient wrt data.

        Parameters
        ----------
        x : CArray
            Pattern with respect to which the gradient will be computed.
            Shape (1, n_features) or (n_features, ).

        Returns
        -------
        gradient : CArray
            Gradient of linear normalizer wrt input data.
            Diagonal matrix of shape (self.w.size, self.w.size).

        """
        # Training first!
        if self.is_clear():
            raise ValueError("train the normalizer first.")

        if x.atleast_2d().shape[1] != self.w.size:
            raise ValueError("input data must have {:} features (columns)."
                             "".format(self.w.size))

        return self.w.diag()
