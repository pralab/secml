"""
.. module:: FeatureReducer
   :synopsis: Common interface for feature reduction algorithms.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from abc import ABCMeta

from secml.ml.features import CPreProcess


class CReducer(CPreProcess):
    """Common interface for feature reduction algorithms."""
    __metaclass__ = ABCMeta
    __super__ = 'CReducer'

    # FIXME: REDUCERS DO NOT SUPPORT THE DEPRECATED clear FRAMEWORK.
    #  REMOVE ALL THE FOLLOWING METHODS AFTER REMOVING THE FRAMEWORK

    def transform(self, x):
        """Apply the transformation algorithm on data.

        Parameters
        ----------
        x : CArray
            Array to be transformed.
            Shape of input array depends on the algorithm itself.

        Returns
        -------
        CArray
            Transformed input data.

        """
        # Transform data using inner preprocess, if defined
        x = self._preprocess_data(x)
        return self._transform(x)

    def revert(self, x):
        """Revert data to original form.

        Parameters
        ----------
        x : CArray
            Transformed array to be reverted to original form.
            Shape of input array depends on the algorithm itself.

        Returns
        -------
        CArray
            Original input data.

        Warnings
        --------
        Reverting a transformed array is not always possible.
        See description of each preprocessor for details.

        """
        v = self._revert(x)

        # Revert data using the inner preprocess, if defined
        if self.preprocess is not None:
            return self.preprocess.revert(v)

        return v

    def gradient(self, x, w=None):
        """Returns the preprocessor gradient wrt data.

        Parameters
        ----------
        x : CArray
            Data array, 2-Dimensional or ravel.
        w : CArray or None, optional
            If CArray, will be left-multiplied to
            the gradient of the preprocessor.

        Returns
        -------
        gradient : CArray
            Gradient of the preprocessor wrt input data.
            If `w` is CArray, will be a vector-like array of size `n_features`.
            Otherwise, the shape depends on the preprocess algorithm.

        """
        out = self._gradient(x, w=w)

        if self.preprocess is not None:
            return self.preprocess.gradient(x, w=out)

        return out
