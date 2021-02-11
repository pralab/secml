"""
.. module:: CClassifierReject
   :synopsis: Interface and common functions for classification with rejection

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from abc import abstractmethod, ABCMeta

from secml.ml.classifiers import CClassifier


class CClassifierReject(CClassifier, metaclass=ABCMeta):
    """Abstract class that defines basic methods for Classifiers with reject.

    A classifier assign a label (class) to new patterns using the
    information learned from training set.

    This interface implements a set of generic methods for training
    and classification that can be used for every algorithms. However,
    all of them can be reimplemented if specific routines are needed.

    Parameters
    ----------
    preprocess : str or CNormalizer
        Features preprocess to applied to input data.
        Can be a CNormalizer subclass or a string with the desired
        preprocess type. If None, input data is used as is.

    """
    __super__ = 'CClassifierReject'

    @abstractmethod
    def predict(self, x, return_decision_function=False, n_jobs=1):
        """Perform classification of each pattern in x.

        If a preprocess has been specified,
        input is normalized before classification.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        return_decision_function : bool, optional
            Whether to return the decision_function value along
            with predictions. Default False.
        n_jobs : int, optional
            Number of parallel workers to use for classification.
            Default 1. Cannot be higher than processor's number of cores.

        Returns
        -------
        labels : CArray
            Flat dense array of shape (n_patterns,) with the label assigned
            to each test pattern. The classification label is the label of
            the class associated with the highest score. The rejected samples
            have label -1.
        scores : CArray, optional
            Array of shape (n_patterns, n_classes) with classification
            score of each test pattern with respect to each training class.
            Will be returned only if `return_decision_function` is True.

        """
        raise NotImplementedError

    def _check_clf_index(self, y):
        """Raise error if index y is outside [-1, n_classes) range.

        Parameters
        ----------
        y : int
            class label index.

        """
        if y < -1 or y >= self.n_classes:
            raise ValueError(
                "class label {:} is out of range".format(y))
