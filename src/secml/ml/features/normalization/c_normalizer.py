"""
.. module:: ArrayNormalizers
   :synopsis: Common interface for array normalizers.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod, abstractproperty

from secml.core import CCreator


class CNormalizer(CCreator):
    """Common interface for normalization preprocessing algorithms.

    Normalization techniques include features scalers, e.g. MinMaxScaler,
    standardizers, e.g. MeanRemover, or features binarizers.

    """
    __metaclass__ = ABCMeta
    __super__ = 'CNormalizer'

    @abstractproperty
    def class_type(self):
        """Defines class type."""
        raise NotImplementedError("the class must define `class_type` "
                                  "attribute to support `CCreator.create()` "
                                  "function properly.")

    def is_linear(self):
        """Returns True for linear normalizers."""
        return False

    @abstractmethod
    def train(self, x):
        """Train normalization algorithm using data.

        Parameters
        ----------
        x : CArray
            Array to be used for training normalization algorithm.
            Shape of input array depends on the algorithm itself.

        Returns
        -------
        CNormalizer
            Instance of the normalization algorithm trained using
            input data.

        """
        raise NotImplementedError(
            "this is an abstract method. Must be overridden in subclass.")

    @abstractmethod
    def normalize(self, x):
        """Apply the normalization algorithm on data.

        Parameters
        ----------
        x : CArray
            Array to be normalized using normalization algorithm.
            Shape of input array depends on the algorithm itself.

        Returns
        -------
        x_transformed : CArray
            Input data normalized using normalization algorithm.

        """
        raise NotImplementedError(
            "this is an abstract method. Must be overridden in subclass.")

    def train_normalize(self, x):
        """Train normalizer using data and then normalize data.

        This method is equivalent to call train(data) and normalize(data)
        in sequence, but it's useful when data is both the training array
        and the array to normalize.

        Parameters
        ----------
        x : CArray
            Array to be normalized using normalization algorithm.
            Each row must correspond to one single patterns, so each
            column is a different feature.

        Returns
        -------
        x_normalized : CArray
            Input data normalized using normalization algorithm.

        See Also
        --------
        train : train the normalizer on input data.
        normalize : normalize input data according training data.

        """
        self.train(x)  # training normalizer first
        return self.normalize(x)

    def revert(self, x):
        """Revert array to original form.

        Parameters
        ----------
        x : CArray
            Normalized array to be reverted to original form.
            Shape of input array depends on the algorithm itself.

        Returns
        -------
        x_original : CArray
            Original input data.

        Notes
        -----
        Reverting a normalized array is not always possible.
        Thus, revert method is not an abstractmethod and should
        be implemented only if applicable.

        """
        raise NotImplementedError(
            "this is a placeholder method. Override if necessary.")

    def gradient(self, x):
        """Returns the normalizer gradient wrt data.

        Parameters
        ----------
        x : CArray
            Data array, 2-Dimensional or ravel.

        Returns
        -------
        gradient : CArray
            Gradient of the normalizer wrt input data.

        """
        raise NotImplementedError("gradient is not implemented for {:}"
                                  "".format(self.__class__.__name__))
