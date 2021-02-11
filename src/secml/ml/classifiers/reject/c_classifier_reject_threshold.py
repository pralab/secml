"""
.. module:: CClassifierRejectThreshold
   :synopsis: Classifier that perform classification with
    rejection based on a defined threshold

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
import math

from secml import _NoValue
from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers import CClassifier
from secml.ml.classifiers.reject import CClassifierReject


class CClassifierRejectThreshold(CClassifierReject):
    """Abstract class that defines basic methods for Classifiers with reject
     based on a certain threshold.

    A classifier assign a label (class) to new patterns using the
    information learned from training set.

    The samples for which the higher score is under a certain threshold are
    rejected by the classifier.

    Parameters
    ----------
    clf : CClassifier
        Classifier to which we would like to apply a reject threshold.
        The classifier can also be already fitted.
    threshold : float
        Rejection threshold.
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    """
    __class_type = 'reject-threshold'

    def __init__(self, clf, threshold, preprocess=None):

        if not isinstance(clf, CClassifier):
            raise ValueError(
                "the inner classifier should be an instance of CClassifier")

        self._clf = clf
        self.threshold = threshold

        super(CClassifierRejectThreshold, self).__init__(preprocess=preprocess)

        if self.clf.is_fitted():
            self._n_features = self._clf.n_features

    @property
    def clf(self):
        """Returns the inner classifier."""
        return self._clf

    @property
    def threshold(self):
        """Returns the rejection threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        """Sets the rejection threshold."""
        self._threshold = float(value)

    @property
    def classes(self):
        """Return the list of classes on which training has been performed."""
        return self._clf.classes.append([-1])

    @property
    def n_classes(self):
        """Number of classes of training dataset, plus the rejection class."""
        return self._clf.n_classes + 1

    def _fit(self, x, y):
        """Private method that trains the One-Vs-All classifier.
        Must be reimplemented by subclasses.

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features).
        y : CArray
            Array of shape (n_samples,) containing the class labels.
        n_jobs : int, optional
            Number of parallel workers to use for training the classifier.
            Default 1. Cannot be higher than processor's number of cores.

        Returns
        -------
        trained_cls : CClassifier
            Instance of the classifier trained using input dataset.

        """
        self._clf.fit(x, y)
        return self

    def _forward(self, x):
        """Private method that computes the decision function.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        rej_scores = CArray.ones(x.shape[0]) * self.threshold
        scores = self._clf.decision_function(x)
        # augment score matrix with reject class scores
        scores = scores.append(rej_scores.T, axis=1)
        return scores

    def predict(self, x, return_decision_function=False, n_jobs=_NoValue):
        """Perform classification of each pattern in x.

        The score matrix of this classifier is equal to the predicted outputs
        plus a column (corresponding to the reject class) with all its values
        equal to :math:`\\theta`, being :math:`\\theta` the reject threshold.

        The predicted class is therefore:

        .. math:: c = \\operatorname*{argmax}_k f_k(x)

        where :math:`c` correspond to the rejection class (i.e., :math:`c=-1`)
        only when the maximum taken over the other classes (excluding the
        reject one) is not greater than the reject threshold :math:`\\theta`.

        If a preprocess has been specified, input is normalized before
        classification.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        return_decision_function : bool, optional
            Whether to return the `decision_function` value along
            with predictions. Default False.
        n_jobs : int, optional
            Number of parallel workers to use for classification.
            Default `_NoValue`. Cannot be higher than processor's
            number of cores.

        Returns
        -------
        labels : CArray
            Flat dense array of shape (n_patterns,) with the label assigned
            to each test pattern. The classification label is the label of
            the class associated with the highest score. The samples for which
            the label is equal -1 are the ones rejected by the classifier
        scores : CArray, optional
            Array of shape (n_patterns, n_classes) with classification
            score of each test pattern with respect to each training class.
            Will be returned only if `return_decision_function` is True.

        """
        if n_jobs is not _NoValue:
            raise ValueError("`n_jobs` is not supported.")

        labels, scores = CClassifier.predict(
            self, x, return_decision_function=True)
        # relabel rejection class
        labels[labels == self.n_classes - 1] = -1
        return (labels, scores) if return_decision_function is True else labels

    def _backward(self, w):
        """Computes the gradient of the classifier's decision function
         wrt decision function input.

        The gradient taken w.r.t. the reject class can be thus set to 0,
        being its output constant regardless of the input sample x.

        Parameters
        ----------
        x : CArray
            The gradient is computed in the neighborhood of x.
        y : int
            Index of the class wrt the gradient must be computed.
            Use -1 to output the gradient w.r.t. the reject class.

        Returns
        -------
        gradient : CArray
            Gradient of the classifier's df wrt its input. Vector-like array.

        """
        # the derivative w.r.t. the rejection class is zero, thus we can just
        # call the clf gradient by removing the last element from w.
        return self.clf.gradient(self._cached_x, w[:-1])

    def compute_threshold(self, rej_percent, ds):
        """Compute the threshold that must be set in the classifier to have
        rej_percent rejection rate (accordingly to an estimation on a
        validation set).

        Parameters
        ----------
        rej_percent : float
            Max percentage of rejected samples.
        ds : CDataset
            Dataset on which the threshold is estimated.

        Returns
        -------
        threshold : float
            The estimated reject threshold
            
        """
        if not self.is_fitted():
            raise NotFittedError("The classifier must be fitted")
        scores = self.predict(ds.X, return_decision_function=True)[1]
        max_scores = scores[:, :-1].max(axis=1).ravel()
        max_scores.sort(inplace=True)
        rej_num = math.floor(rej_percent * ds.num_samples)
        threshold = max_scores[rej_num - 1].item()
        self.logger.info("Chosen threshold: {:}".format(threshold))
        return threshold
