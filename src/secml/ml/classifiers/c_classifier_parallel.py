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




    # def predict(self, x, return_decision_function=False, n_jobs=1):
    #
    #     x = x.atleast_2d()  # Ensuring input is 2-D
    #
    #     scores = CArray.ones(shape=(x.shape[0], self.n_classes))
    #
    #     # Compute the decision function for each training class in parallel
    #     res = parfor2(_classify_one, self.n_classes,
    #                   n_jobs, self, x, self.verbose)
    #
    #     # Build results array by extracting the scores for each training class
    #     for i in range(self.n_classes):
    #         scores[:, i] = CArray(res[i]).T
    #
    #     # The classification label is the label of the class
    #     # associated with the highest score
    #     labels = scores.argmax(axis=1).ravel()
    #
    #     return (labels, scores) if return_decision_function is True else labels
