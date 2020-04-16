"""
.. module:: CPreProcess
   :synopsis: Common interface for feature preprocessing algorithms.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta, abstractmethod
from secml.ml import CModule


class CPreProcess(CModule, metaclass=ABCMeta):
    """Common interface for feature preprocessing algorithms.

    Parameters
    ----------
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    """
    __super__ = 'CPreProcess'

    def __init__(self, preprocess=None):
        CModule.__init__(self, preprocess=preprocess)

    def fit(self, x, y=None):
        """Fit the preprocessor.

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
        CPreProcess
            Instance of the trained normalizer.

        """
        return CModule.fit(self, x, y)

    def fit_transform(self, x, y=None):
        """Fit preprocessor using data and then transform data.

        This method is equivalent to call fit(data) and transform(data)
        in sequence, but it's useful when data is both the training array
        and the array to be transformed.

        Parameters
        ----------
        x : CArray
            Array to be transformed.
            Each row must correspond to one single patterns, so each
            column is a different feature.
        y : CArray or None, optional
            Flat array with the label of each pattern.
            Can be None if not required by the preprocessing algorithm.

        Returns
        -------
        CArray
            Transformed input data.

        See Also
        --------
        fit : fit the preprocessor.
        transform : transform input data.

        """
        return self.fit_forward(x=x, y=y, caching=False)

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
        return self.forward(x, caching=False)

    def _inverse_transform(self, x):
        raise NotImplementedError(
            "inverting this transformation is not supported.")

    def inverse_transform(self, x):
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
        self._check_is_fitted()

        v = self._inverse_transform(x)

        # Revert data using the inner preprocess, if defined
        if self.preprocess is not None:
            return self.preprocess.inverse_transform(v)

        return v

    # Same doc for the protected method
    _inverse_transform.__doc__ = inverse_transform.__doc__
