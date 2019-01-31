"""
.. module:: CClassifierRejectThreshold
   :synopsis: Classifier that perform classification with
    rejection based on a defined threshold

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml import _NoValue
from secml.array import CArray
from secml.ml.classifiers import CClassifier
from secml.ml.classifiers.reject import CClassifierReject


class CClassifierRejectThreshold(CClassifierReject):
    """Abstract class that defines basic methods for Classifiers with reject
     based on a certain threshold.

    A classifier assign a label (class) to new patterns using the
    informations learned from training set.

    The samples for which the higher score is under a certain threshold are
    rejected by the classifier.

    Parameters
    ----------
    clf : CClassifier
        Classifier to which we would like to apply a reject threshold.
    threshold : float
        Rejection threshold.

    """
    __class_type = 'reject-threshold'

    def __init__(self, clf, threshold):

        self.clf = clf
        self.threshold = threshold

        super(CClassifierRejectThreshold, self).__init__()

    def __clear(self):
        """Reset the object."""
        self._clf.clear()

    def __is_clear(self):
        """Returns True if object is clear."""
        return self._clf.is_clear()

    @property
    def clf(self):
        """Returns the inner classifier."""
        return self._clf

    @clf.setter
    def clf(self, value):
        """Sets the inner classifier."""
        if isinstance(value, CClassifier):
            self._clf = value
        else:
            raise ValueError(
                "the inner classifier should be an istance of CClassifier")

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
        return self._clf.classes

    @property
    def n_classes(self):
        """Number of classes of training dataset."""
        return self._clf.n_classes

    @property
    def n_features(self):
        """Number of features"""
        return self._clf.n_features

    def fit(self, dataset, n_jobs=1):
        """Trains the classifier.

        If a preprocess has been specified,
        input is normalized before training.

        For multiclass case see `.CClassifierMulticlass`.

        Parameters
        ----------
        dataset : CDataset
            Training set. Must be a :class:`.CDataset` instance with
            patterns data and corresponding labels.
        n_jobs : int, optional
            Number of parallel workers to use for training the classifier.
            Default 1. Cannot be higher than processor's number of cores.

        Returns
        -------
        trained_cls : CClassifier
            Instance of the classifier trained using input dataset.

        """
        return self._fit(dataset, n_jobs)

    def _fit(self, dataset, n_jobs=1):
        """Private method that trains the One-Vs-All classifier.
        Must be reimplemented by subclasses.

        Parameters
        ----------
        dataset : CDataset
            Training set. Must be a :class:`.CDataset` instance with
            patterns data and corresponding labels.
        n_jobs : int, optional
            Number of parallel workers to use for training the classifier.
            Default 1. Cannot be higher than processor's number of cores.

        Returns
        -------
        trained_cls : CClassifier
            Instance of the classifier trained using input dataset.

        """
        return self._clf.fit(dataset, n_jobs=n_jobs)

    def decision_function(self, x, y):
        """Computes the decision function for each pattern in x.

        If a preprocess has been specified, input is normalized
        before computing the decision function.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : int
            Index of the class wrt the gradient must be computed, -1 to
            compute it w.r.t. the reject class

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        return self._decision_function(x, y)

    def _decision_function(self, x, y):
        """Private method that computes the decision function.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : int
            Index of the class wrt the gradient must be computed, -1 to
            compute it w.r.t. the reject class

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        if y == -1:
            # compute the score of the predicted class
            pred_labels, scores = self._clf.predict(
                x, return_decision_function=True)

            # compute the score of the reject class
            max_scores = scores.max(axis=1).ravel()

            # return -1 * the score of the predicted class
            return -max_scores

        elif y < self.n_classes:

            return self._clf.decision_function(x, y)

        else:
            raise ValueError("The index of the class wrt the decision "
                             "function must be computed is wrong.")

    def predict(self, x, return_decision_function=False, n_jobs=_NoValue):
        """Perform classification of each pattern in x.

        If a preprocess has been specified,
         input is normalized before classification.

        The samples for which the higher score is lower than a certain
        threshold are rejected by the classifier

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
            Default _NoValue. Cannot be higher than processor's number of cores.

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

        labels, scores = self._clf.predict(x, return_decision_function=True)

        # Apply reject :
        # compute the score of the reject class
        rej_scores = self.decision_function(x, y=-1).T

        # find the maximum score
        scores_max = scores.max(axis=1)

        # Assign -1 to rejected sample labels
        labels[CArray(scores_max.ravel() <= self.threshold).ravel()] = -1

        # Return the expected type for labels (CArray)
        labels = labels.ravel()

        # augment score matrix with reject class scores
        scores = scores.append(rej_scores, axis=1)

        return (labels, scores) if return_decision_function is True else labels

    def _gradient_f(self, x, y):
        """Computes the gradient of the classifier's decision function
         wrt decision function input.

        Parameters
        ----------
        x : CArray
            The gradient is computed in the neighborhood of x.
        y : int
            Index of the class wrt the gradient must be computed, -1 to
            have the gradient w.r.t. the reject class

        Returns
        -------
        gradient : CArray
            Gradient of the classifier's df wrt its input. Vector-like array.

        """
        if y == -1:

            # Compute the gradient w.r.t. the reject class, which is -1 *
            # the gradient of the predicted class.

            # find the predicted class
            label, score = self._clf.predict(x, return_decision_function=True)

            # return -1 * the gradient of the predicted class
            return -self._clf.gradient_f_x(x, y=label.item())

        elif y < self.n_classes:
            return self._clf.gradient_f_x(x, y=y)

        else:
            raise ValueError("The index of the class wrt the gradient must "
                             "be computed is wrong.")
