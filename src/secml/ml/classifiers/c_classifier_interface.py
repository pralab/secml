"""
.. module:: CClassifier
   :synopsis: Interface and common functions for classification

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod
import six
from six.moves import range

from secml.ml.classifiers import CClassifier
from secml.array import CArray
from secml.data import CDataset
from secml.ml.features import CPreProcess
from secml.parallel import parfor2
from secml.utils.mixed_utils import check_is_fitted
from secml.core.exceptions import NotFittedError


def _classify_one(tr_class_idx, clf, test_x, verbose):
    """Performs classification wrt class of label `tr_class_idx`.

    Parameters
    ----------
    tr_class_idx : int
        Index of the label against which the classifier should be trained.
    clf : CClassifierInterface
        Instance of the classifier.
    test_x : CArray
        Test data as 2D CArray.
    verbose : int
        Verbosity level of the logger.

    """
    # Resetting verbosity level. This is needed as objects
    # change id  when passed to subprocesses and our logging
    # level is stored per-object looking to id
    clf.verbose = verbose
    # Getting predicted data for current class classifier
    return clf.decision_function(test_x, y=tr_class_idx)


@six.add_metaclass(ABCMeta)
class CClassifierInterface(CClassifier):
    """Abstract class that defines basic methods for Classifiers.

    A classifier assign a label (class) to new patterns using the
    informations learned from training set.

    This interface implements a set of generic methods for training
    and classification that can be used for every algorithms. However,
    all of them can be reimplemented if specific routines are needed.

    Parameters
    ----------
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    """
    __super__ = 'CClassifierInterface'

    def __init__(self, preprocess=None):
        CClassifier.__init__(self, preprocess=preprocess)

    @abstractmethod
    def _decision_function(self, x, y):
        """Private method that computes the decision function.

        .. warning:: Must be reimplemented by a subclass of `.CClassifier`.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : int
            The label of the class wrt the function should be calculated.

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        raise NotImplementedError()

    def decision_function(self, x, y):
        """Computes the decision function for each pattern in x.

        If a preprocess has been specified, input is normalized
        before computing the decision function.

        .. note::

            The actual decision function should be implemented
            case by case inside :meth:`_decision_function` method.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : int
            The label of the class wrt the function should be calculated.

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_patterns,).

        Warnings
        --------
        This method implements a generic formulation where the
         decision function is computed separately for each pattern.
         It's convenient to override this when the function can be computed
         for all patterns at once to improve performance.

        """
        self._check_is_fitted()

        x = x.atleast_2d()  # Ensuring input is 2-D

        # Transform data if a preprocess is defined
        x = self._preprocess_data(x)

        score = CArray.ones(shape=x.shape[0])
        for i in range(x.shape[0]):
            score[i] = self._decision_function(x[i, :], y)

        return score

    def predict(self, x, return_decision_function=False, n_jobs=1):
        """Perform classification of each pattern in x.

        If a preprocess has been specified,
         input is normalized before classification.

        Parameters
        ----------
        return_decision_function
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
             the class associated with the highest score.
        scores : CArray, optional
            Array of shape (n_patterns, n_classes) with classification
             score of each test pattern with respect to each training class.
            Will be returned only if `return_decision_function` is True.

        Warnings
        --------
        This method implements a generic formulation where the
         classification score is computed separately for training class.
         It's convenient to override this when the score can be computed
         for one of the classes only, e.g. for binary classifiers the score
         for the positive/negative class is commonly the negative of the
         score of the other class.

        """
        x = x.atleast_2d()  # Ensuring input is 2-D

        scores = CArray.ones(shape=(x.shape[0], self.n_classes))

        # Compute the decision function for each training class in parallel
        res = parfor2(_classify_one, self.n_classes,
                      n_jobs, self, x, self.verbose)

        # Build results array by extracting the scores for each training class
        for i in range(self.n_classes):
            scores[:, i] = CArray(res[i]).T

        # The classification label is the label of the class
        # associated with the highest score
        labels = scores.argmax(axis=1).ravel()

        return (labels, scores) if return_decision_function is True else labels
