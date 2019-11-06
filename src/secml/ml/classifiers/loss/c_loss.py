"""
.. module:: CLoss
   :synopsis: Interface for Loss Functions

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from abc import ABCMeta, abstractmethod

from secml.core import CCreator
from secml.array import CArray


class CLoss(CCreator, metaclass=ABCMeta):
    """Interface for loss functions."""
    __super__ = 'CLoss'

    @property
    @abstractmethod
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
            2-D array of shape (n_samples, n_classes) or
            1-D flat array of shape (n_samples,).
        pos_label : int or None, optional
            Default None, meaning that the function is computed
            for each sample wrt the corresponding true label.
            Otherwise, this is the class wrt compute the loss function.
            If `score` is a 1-D flat array, this parameter is ignored.

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
            2-D array of shape (n_samples, n_classes) or 1-D flat array
            of shape (n_samples,).
        pos_label : int or None, optional
            Default None, meaning that the function derivative is computed
            for each sample wrt the corresponding true label.
            Otherwise, this is the class wrt compute the derivative.
            If `score` is a 1-D flat array, this parameter is ignored.

        Returns
        -------
        CArray
            Derivative of the loss function. Vector-like array.

        """
        raise NotImplementedError()


def _check_binary_score(score, pos_label=1):
    """Check that input scores are binary and return desired column.

    If score is a 1-D flat array, the probabilities are returned as is.
    If score is 2-D (n_samples, n_classes), it is checked to be
    binary (2-classes) and the column corresponding to pos_label is returned.

    Parameters
    ----------
    score : CArray
        Outputs (predicted), targets.
        2-D array of shape (n_samples, n_classes) or
        1-D flat array of shape (n_samples,).
    pos_label : {0, 1}, optional
        The index of the column to return. Default 1.
        If `score` is a 1-D flat array, this parameter is ignored.

    Returns
    -------
    CArray
        Scores. Vector-like array.

    """
    if score.ndim == 2:
        if score.shape[1] > 2:
            raise ValueError(
                "only 2 classes are supported. "
                "`score` has shape[1] = {:}".format(score.shape[1]))
        else:
            score = score[:, pos_label].ravel()

    return score  # Manage 1-D/2-D case
