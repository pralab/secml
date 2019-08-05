"""
.. module:: CClassifierRejectThreshold
   :synopsis: Classifier that perform classification with
    rejection based on a defined threshold

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml import _NoValue
from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers import CClassifier
from secml.ml.classifiers.reject import CClassifierReject
from secml.ml.classifiers.reject.mixin_classifier_gradient_reject_threshold import \
    CClassifierGradientRejectThresholdMixin


class CClassifierRejectThreshold(CClassifierReject,
                                 CClassifierGradientRejectThresholdMixin):
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
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    """
    __class_type = 'reject-threshold'

    def __init__(self, clf, threshold, preprocess=None):

        self.clf = clf
        self.threshold = threshold

        if self.clf.preprocess is not None:
            raise ValueError(
                "the preprocessor should be passed to the outer classifier.")

        super(CClassifierRejectThreshold, self).__init__(preprocess=preprocess)

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
                "the inner classifier should be an instance of CClassifier")

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

    def fit(self, dataset, n_jobs=1):
        """Trains the classifier.

        If a preprocess has been specified,
        input is normalized before training.

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
        self._n_features = dataset.num_features

        data_x = dataset.X
        # Transform data if a preprocess is defined
        if self.preprocess is not None:
            data_x = self.preprocess.fit_transform(dataset.X)

        return self._fit(CDataset(data_x, dataset.Y), n_jobs=n_jobs)

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
        self._clf.fit(dataset, n_jobs=n_jobs)
        return self

    def decision_function(self, x, y):
        """Computes the decision function for each pattern in x.

        The discriminant function of the reject class is a vector with all its
        values equal to :math:`\theta`, being :math:`\theta` the reject
        threshold.

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
        self._check_is_fitted()

        x = x.atleast_2d()  # Ensuring input is 2-D

        # Transform data if a preprocess is defined
        x = self._preprocess_data(x)

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
        x = x.atleast_2d()

        if y == -1:
            # the score of the reject class is a vector with all the elements
            # equals to the reject threshold
            return CArray.ones(x.shape[0]) * self.threshold

        elif y < self.n_classes:
            return self._clf.decision_function(x, y)

        else:
            raise ValueError("The index of the class wrt the decision "
                             "function must be computed is wrong.")

    def predict(self, x, return_decision_function=False, n_jobs=_NoValue):
        """Perform classification of each pattern in x.

        The score matrix of this classifier is equal to the predicted outputs
        plus a column (corresponding to the reject class) with all its values
        equal to :math:`\\theta`, being :math:`\\theta` the reject threshold.

        The predicted class is therefore:

        .. math:: c = \\operatorname*{argmin}_k f_k(x)

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

        x_in = x  # Original data

        # Transform data if a preprocess is defined
        x = self._preprocess_data(x)

        labels, scores = self._clf.predict(x, return_decision_function=True)

        # Apply reject

        # compute the score of the reject class
        rej_scores = self.decision_function(x_in, y=-1).T

        # find the maximum score
        scores_max = scores.max(axis=1)

        # Assign -1 to rejected sample labels
        labels[CArray(scores_max.ravel() < self.threshold).ravel()] = -1

        # Return the expected type for labels (CArray)
        labels = labels.ravel()

        # augment score matrix with reject class scores
        scores = scores.append(rej_scores, axis=1)

        return (labels, scores) if return_decision_function is True else labels
