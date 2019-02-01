"""
.. module:: CClassifierRejectDetector
   :synopsis: Classifier that detect adversarial samples using a detector

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml import _NoValue
from secml.array import CArray
from secml.data.c_dataset import CDataset
from secml.ml.classifiers import CClassifier
from secml.ml.classifiers.loss import CSoftmax
from secml.ml.classifiers.reject import CClassifierReject


class CClassifierRejectDetector(CClassifierReject):
    """Classifier with reject based on detector.

    Classifier that rejects the evasion samples based on the score
    assigned to them by another classifier (detector) that is trained
    to distinguish between legitimate and adversarial samples.

    Parameters
    ----------
    clf : CClassifier
        Classifier to which we would like to apply the defense.
    det : CClassifier
        Binary classifier that will be trained to detect the adversarial
        samples.
    adv_x : CArray
        Array containing already computed adversarial samples.

    """
    __class_type = 'reject-detector'

    def __init__(self, clf, det, adv_x):

        if not isinstance(clf, CClassifier):
            raise TypeError("`clf` must be an instance of `CClassifier`")
        self._clf = clf
        if not isinstance(det, CClassifier):
            raise TypeError("`det` must be an instance of `CClassifier`")
        self._det = det

        self.adv_x = adv_x

        super(CClassifierRejectDetector, self).__init__()

    def __clear(self):
        """Reset the object."""
        self._det.clear()
        self._clf.clear()

    def __is_clear(self):
        """Returns True if object is clear."""
        return self._det.is_clear() and self._clf.is_clear()

    @property
    def clf(self):
        """Return the inner classifier."""
        return self._clf

    @property
    def det(self):
        """Return the inner detector."""
        return self._det

    @property
    def adv_x(self):
        """Array containing already computed adversarial samples."""
        return self._adv_x

    @adv_x.setter
    def adv_x(self, value):
        """Array containing already computed adversarial samples."""
        self._adv_x = value.atleast_2d()

    def _normalize_scores(self, orig_score):
        """Normalizes the scores using softmax."""
        return CSoftmax().softmax(orig_score)

    def fit(self, dataset, n_jobs=1):
        """Trains both the classifier and the detector.

        Parameters
        ----------
        dataset : CDataset
            Training set. Must be a :class:`.CDataset` instance with
            patterns data and corresponding labels.
        n_jobs : int
            Number of parallel workers to use for training the classifier.
            Default 1. Cannot be higher than processor's number of cores.

        Returns
        -------
        trained_cls : CClassifier
            Instance of the classifier trained using input dataset.

        """
        if not isinstance(dataset, CDataset):
            raise TypeError(
                "training set should be provided as a CDataset object.")

        # Resetting the classifier
        self.clear()

        # Storing dataset classes
        self._classes = dataset.classes
        self._n_features = dataset.num_features

        data_x = dataset.X
        # Preprocessing data if a preprocess is defined
        if self.preprocess is not None:
            data_x = self.preprocess.fit_normalize(dataset.X)

        return self._fit(CDataset(data_x, dataset.Y), n_jobs=n_jobs)

    def _fit(self, dataset, n_jobs=1):
        """Trains both the classifier and the detector.

        Parameters
        ----------
        dataset : CDataset
            Training set. Must be a :class:`.CDataset` instance with
            patterns data and corresponding labels.
        n_jobs : int
            Number of parallel workers to use for training the classifier.
            Default 1. Cannot be higher than processor's number of cores.

        Returns
        -------
        trained_cls : CClassifier
            Instance of the classifier trained using input dataset.

        """
        self._clf.fit(dataset, n_jobs=n_jobs)

        # create a dataset where the negative samples are all the dataset
        # samples and the negative one are the adversarial samples

        # store the dataset samples as positive class
        dataset = dataset.deepcopy()
        dataset.Y[:] = 0
        adv_x = self.adv_x
        n_adv_x = adv_x.shape[0]
        adv_dts = CDataset(adv_x, CArray.ones(n_adv_x))

        dataset = dataset.append(adv_dts)

        self._det.fit(dataset, n_jobs=n_jobs)

        # use scores on a point to check that the detector is binary
        det_scores = self._det.predict(
            adv_dts.X[0, :], return_decision_function=True)[1]
        self._check_det_scores(det_scores)

        return self

    def decision_function(self, x, y):
        """Computes the decision function for each pattern in x.

        The scores of the classifier are normalized
        The score of the reject class (y = -1) is the normalized score of the
        detector

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
        if self.is_clear():
            raise ValueError("make sure the classifier is trained first.")

        return self._decision_function(x, y)

    def _decision_function(self, x, y):
        """Private method that computes the decision function.

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
        self.logger.info(
            "Getting decision function against class: {:}".format(y))

        if y == -1:
            # get the detector decision function
            scores = self._det.predict(x, return_decision_function=True)[1]
            # normalize the scores
            return (self._normalize_scores(scores)[:, 1]).ravel()

        elif y < self.n_classes:

            # we call the predict to have the score w.r.t. all the classes
            scores = self._clf.predict(x, return_decision_function=True)[1]
            # return the score of the required class
            return (self._normalize_scores(scores)[:, y]).ravel()

        else:
            raise ValueError("The index of the class wrt the gradient must "
                             "be computed is wrong.")

    @staticmethod
    def _check_det_scores(det_scores):
        """Check detector scores have two classes."""
        det_scores = det_scores.atleast_2d()
        if not det_scores.shape[1] == 2:
            raise ValueError("The detector should be a binary classifier")

    def predict(self, x, return_decision_function=False, n_jobs=_NoValue):
        """Perform classification of each pattern in x.

        The samples which are classified as malicious by the detector are
        rejected

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        return_decision_function : bool, optional
            Whether to return the decision_function value along
            with predictions. Default False.

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
             An additional column for the reject class is added to the
             classifier scores
            Will be returned only if `return_decision_function` is True.

        """
        if n_jobs is not _NoValue:
            raise ValueError("`n_jobs` is not supported.")

        # detector prediction
        det_pred, det_scores = self._det.predict(
            x, return_decision_function=True)

        # classifier prediction
        clf_labels, clf_scores = self._clf.predict(
            x, return_decision_function=True)

        clf_labels, clf_scores = self._reject(
            clf_labels, clf_scores, det_pred, det_scores)

        return (clf_labels, clf_scores) if \
            return_decision_function is True else clf_labels

    def _reject(self, labels, scores, det_pred, det_scores):
        """Rejects samples that are classified as adversarial from the detector.

        An additional column for the reject class is added to the classifier
        scores (the score of the reject class is the detector score)

        """
        # Ensure we work with CArrays
        labels = CArray(labels)
        scores = CArray(scores)

        scores = self._normalize_scores(scores)
        rej_scores = self._normalize_scores(det_scores)[:, 1]

        # augment score matrix with reject region score
        scores = scores.append(rej_scores, axis=1)

        # Assign -1 to rejected sample labels
        labels[CArray(det_pred.ravel() == 1).ravel()] = -1

        # Return the expected type for labels (CArray)
        labels = labels.ravel()

        return labels, scores

    def _gradient_f(self, x, y):
        """Computes the gradient of the classifier's decision function
         wrt decision function input.

        Parameters
        ----------
        x : CArray
            The gradient is computed in the neighborhood of x.
        y : int
            Index of the class wrt the gradient must be computed, -1 if to
            have the gradient w.r.t. the reject class

        Returns
        -------
        gradient : CArray
            Gradient of the classifier's df wrt its input. Vector-like array.

        """
        if y == -1:
            # return the gradient of the detector
            # (it's binary so always return y=1)
            grad = self._det.gradient_f_x(x, y=1)

        elif y < self.n_classes:
            grad = self._clf.gradient_f_x(x, y=y)

        else:
            raise ValueError("The index of the class wrt the gradient must "
                             "be computed is wrong.")

        return grad.ravel()
