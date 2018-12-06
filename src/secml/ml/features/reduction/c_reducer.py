"""
.. module:: ArrayReduction
   :synopsis: Common interface for matrix reduction algorithms.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod, abstractproperty

from secml.core import CCreator


class CReducer(CCreator):
    """Common interface for matrix reduction algorithms.

    Most of the reduction algorithms, such as PCA or LDA, can
    be regarded as dimensionality reduction techniques.

    """
    __metaclass__ = ABCMeta
    __super__ = 'CReducer'

    @abstractproperty
    def class_type(self):
        """Defines class type."""
        raise NotImplementedError("the class must define `class_type` "
                                  "attribute to support `CCreator.create()` "
                                  "function properly.")

    @abstractmethod
    def train(self, data, y):
        """Train reduction algorithm using data.

        Parameters
        ----------
        data : CArray
            Array to be used for training reduction algorithm.
            Shape of input array depends on the algorithm itself.
        y : CArray
            Flat Carray with target values. This is not used by all
            reduction algorithms.

        Returns
        -------
        self_trained : CReducer
            Instance of the reduction algorithm trained using
            input data.

        """
        return NotImplementedError("this is an abstract method. Must be overriden in subclass.")

    @abstractmethod
    def transform(self, data):
        """Apply the reduction algorithm on data.

        Parameters
        ----------
        data : CArray
            Array to be transformed using reduction algorithm.
            Shape of input array depends on the algorithm itself.

        Returns
        -------
        data_transformed : CArray
            Input data transformed using reduction algorithm.

        """
        return NotImplementedError("this is an abstract method. Must be overriden in subclass.")

    def train_transform(self, data, y=None):
        """Train reduction algorithm using data and then transform data.

        This method is equivalent to call train(data) and transform(data)
        in sequence, but it's useful when data is both the training array
        and the array to transform.

        Parameters
        ----------
        data : CArray
            Array to be transformed using reduction algorithm.
            Each row must correspond to one single pattern, so each
            column is a different feature.
        y : CArray
            Flat Carray with target values. This is not used by all
            reduction algorithms.

        Returns
        -------
        data_transformed : CArray
            Input data transformed using reduction algorithm.

        See Also
        --------
        train : train the reduction algorithm on input data.
        transform : transform input data according training data.

        """
        self.train(data, y)  # training reduction first
        return self.transform(data)

    def revert(self, data):
        """Revert array to original form.

        Parameters
        ----------
        data : CArray
            Transformed array to be reverted to original form.
            Shape of input array depends on the algorithm itself.

        Returns
        -------
        data_original : CArray
            Original input data.

        Notes
        -----
        Reverting a transformed array is not always possible.
        Thus, revert method is not an abstractmethod and should
        be implemented only if applicable.

        """
        return NotImplementedError("this is a placeholder method. Override if necessary.")
