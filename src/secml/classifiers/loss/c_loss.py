"""
.. module:: CLoss
   :synopsis: Interface for Loss Functions

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod, abstractproperty

from secml.core import CCreator
from secml.array import CArray


class CLoss(CCreator):
    """Interface for loss functions."""
    __metaclass__ = ABCMeta
    __super__ = 'CLoss'

    @abstractproperty
    def class_type(self):
        """Defines loss function type."""
        raise NotImplementedError()

    @abstractproperty
    def suitable_for(self):
        """Defines which problem the loss is suitable for.

        Accepted values:
        - classification
        - regression

        """
        raise NotImplementedError()

    @abstractmethod
    def loss(self, y_true, score):
        """Computes the value of the loss function.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.

        Returns
        -------
        CArray
            Loss function. Vector-like array.

        """
        raise NotImplementedError()

    def dloss(self, y_true, score):
        """Computes the derivative of the loss function with respect to `score`.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.

        Returns
        -------
        CArray
            Derivative of the loss function. Vector-like array.

        """
        raise NotImplementedError()


class CLossRegression(CLoss):
    """Interface for loss functions suitable for regression problems."""
    suitable_for = 'regression'

    @abstractmethod
    def loss(self, y_true, score):
        """Computes the value of the loss function.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            Vector-like array of shape (n_samples,).

        Returns
        -------
        CArray
            Loss function. Vector-like array.

        """
        raise NotImplementedError()

    @abstractmethod
    def dloss(self, y_true, score):
        """Computes the derivative of the loss function with respect to `score`.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            Vector-like array of shape (n_samples,).

        Returns
        -------
        CArray
            Derivative of the loss function. Vector-like array.

        """
        raise NotImplementedError()


class CLossClassification(CLoss):
    """Interface for loss functions suitable for classification problems."""
    suitable_for = 'classification'

    @abstractmethod
    def loss(self, y_true, score, pos_label=None):
        """Computes the value of the loss function.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            2-D array of shape (n_samples, n_classes) or vector-like
             array of shape (n_samples,).
        pos_label : int or None, optional
            Default None, meaning that the function is computed
             for each sample wrt the corresponding true label.
            Otherwise, this is the class wrt compute the loss function.
            If `score` is vector-like, this parameter is ignored.

        Returns
        -------
        CArray
            Loss function. Vector-like array.

        """
        raise NotImplementedError()

    def dloss(self, y_true, score, pos_label=None):
        """Computes the derivative of the loss function with respect to `score`.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            2-D array of shape (n_samples, n_classes) or vector-like
             array of shape (n_samples,).
        pos_label : int or None, optional
            Default None, meaning that the function derivative is computed
             for each sample wrt the corresponding true label.
            Otherwise, this is the class wrt compute the derivative.
            If `score` is vector-like, this parameter is ignored.

        Returns
        -------
        CArray
            Derivative of the loss function. Vector-like array.

        """
        raise NotImplementedError()
