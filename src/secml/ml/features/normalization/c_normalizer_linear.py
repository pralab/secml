"""
.. module:: CNormalizerLinear
   :synopsis: Interface for linear normalizers.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import abstractmethod

from secml.core.decorators import deprecated
from secml.array import CArray
from secml.ml.features.normalization import CNormalizer
from secml.utils.mixed_utils import check_is_fitted


# TODO: ADD SPARSE ARRAYS SUPPORT
class CNormalizerLinear(CNormalizer):
    """Standardizes array by linearly scaling each feature.

    Input data must have one row for each patterns,
    so features to scale are on each array's column.

    The standardization is given by::

        X_scaled = m * X(axis=0) + q

    where m, q are specific constants for each normalization.

    Notes
    -----
    Differently from numpy, we manage flat vectors as 2-Dimensional of
    shape (1, array.size). This means that normalizing a flat vector is
    equivalent to transform array.atleast_2d(). To obtain a numpy-style
    normalization of flat vectors, transpose array first.

    """

    @property
    @abstractmethod
    def w(self):
        """Returns the step of the linear normalizer."""
        # w must be a CArray
        raise NotImplementedError("Linear normalizer should define the slope.")

    @property
    @abstractmethod
    def b(self):
        """Returns the bias of the linear normalizer."""
        # b must be a CArray
        raise NotImplementedError("Linear normalizer should define the bias.")

    def _check_is_fitted(self):
        """Check if the preprocessor is trained (fitted).

        Raises
        ------
        NotFittedError
            If the preprocessor is not fitted.

        """
        check_is_fitted(self, ['w', 'b'])

    def _forward(self, x):
        """Linearly scales features.

        Parameters
        ----------
        x : CArray
            Array with features to be scaled. Must have the same number
            of features (i.e. the number of columns) of training array.

        Returns
        -------
        Array with features linearly scaled.
        Shape of returned array is the same of the original array.

        """
        if x.atleast_2d().shape[1] != self.w.size:
            raise ValueError("array to normalize must have {:} "
                             "features (columns).".format(self.w.size))

        return (x * self.w).todense() + self.b

    def _inverse_transform(self, x):
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
        if x.atleast_2d().shape[1] != self.w.size:
            raise ValueError("array to revert must have {:} "
                             "features (columns).".format(self.w.size))

        v = (x - self.b).atleast_2d()

        v[:, self.w != 0] /= self.w[self.w != 0]  # avoids division by zero

        return v.ravel() if x.ndim <= 1 else v

    def _backward(self, w=None):
        """Compute the gradient wrt the cached inputs during the forward pass.

        Parameters
        ----------
        w : CArray or None, optional
            If CArray, will be left-multiplied to the gradient
            of the preprocessor.

        Returns
        -------
        gradient : CArray
            Gradient of the linear normalizer wrt input data.
            - a flat array of shape (x.shape[1], ) if `w` is None;
            - if `w` is passed as input, will have (w.shape[0], x.shape[1]),
              or (x.shape[1], ) if `w` is a flat array.

        """
        grad = self.w  # Should be I * self.w . We keep a vector for simplicity

        # Left multiply input `w` with normalizer gradient
        return w * grad if w is not None else grad
