"""
.. module:: ClassifierLinear
   :synopsis: Interface and common functions for Linear Classifiers

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml.classifiers import CClassifier
from secml.classifiers.clf_utils import extend_binary_labels
from secml.array import CArray
from secml.data import CDataset


class CClassifierLinear(CClassifier):
    """Abstract class that defines basic methods for linear classifiers.

    A linear classifier assign a label (class) to new patterns
    computing the inner product between the patterns and a vector
    of weights for each training set feature.

    This interface implements a set of generic methods for training
    and classification that can be used for every linear model.

    Parameters
    ----------
    normalizer : str, CNormalizer
        Features normalizer to applied to input data.
        Can be a CNormalizer subclass or a string with the desired
        normalizer type. If None, input data is used as is.

    """

    def __init__(self, normalizer=None):
        # Linear classifier parameters
        self._w = None
        self._b = None

        # Calling init of CClassifier
        CClassifier.__init__(self, normalizer=normalizer)

    def __clear(self):
        """Reset the object."""
        self._w = None
        self._b = None

    @property
    def w(self):
        """Vector with each feature's weight (dense or sparse)."""
        return self._w

    @property
    def b(self):
        """Bias calculated from training data."""
        return self._b

    def is_linear(self):
        """Return True as the classifier is linear."""
        if self.normalizer is None or \
                self.normalizer is not None and self.normalizer.is_linear():
            return True
        return False

    def is_clear(self):
        """Returns True if object is clear."""
        return super(CClassifierLinear, self).is_clear() and \
            self._w is None and self._b is None

    def train(self, dataset, n_jobs=1):
        """Trains the linear classifier.

        If a normalizer has been specified,
        input is normalized before training.

        Training on 2nd class is avoided to speed up classification.

        Parameters
        ----------
        dataset : CDataset
            Binary (2-classes) training set. Must be a :class:`.CDataset`
            instance with patterns data and corresponding labels.
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
                "training set should be provided as a single dataset.")
        if dataset.num_classes != 2:
            raise ValueError(
                "training available on binary (2-classes) datasets only.")

        return super(CClassifierLinear, self).train(dataset, n_jobs=n_jobs)

    def _discriminant_function(self, x, label=1):
        """Compute the distance of the samples in x from the separating hyperplane.

        Discriminant function is always computed wrt positive class.

        Parameters
        ----------
        x : CArray or array_like
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).

        Returns
        -------
        score : CArray or scalar
            Flat array of shape (n_patterns,) with discriminant function
            value of each test pattern or scalar if n_patterns == 1.

        """
        if self.is_clear():
            raise ValueError("make sure the classifier is trained first.")
        if label != 1:
            raise ValueError(
                "discriminant function is always computed wrt positive class.")
        # Computing: `x * w^T`
        return CArray(x.dot(self.w.T)).todense().ravel() + self.b

    def discriminant_function(self, x, label=1):
        """Computes the discriminant function for each pattern in x.

        If a normalizer has been specified, input is normalized
        before computing the discriminant function.

        For a linear classifier the discriminant function is given by::

            f[i] =  (x[i] * w^T) + b

        Parameters
        ----------
        x : CArray or array_like
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        label : int
            The label of the class with respect to which the function
            should be calculated. Default 1.

        Returns
        -------
        score : CArray or scalar
            Flat array of shape (n_patterns,) or scalar if
            `n_patterns == 1` with the value of the discriminant
            function for each test pattern.

        """
        if self.is_clear():
            raise ValueError("make sure the classifier is trained first.")
        x_carray = CArray(x).atleast_2d()

        # Normalizing data if a normalizer is defined
        if self.normalizer is not None:
            x_carray = self.normalizer.normalize(x_carray)

        # We have a binary classifier
        sign = 2 * label - 1  # Sign depends on input label (0/1)

        # Return a scalar if n_patterns == 1
        score = sign * self._discriminant_function(x_carray).ravel()
        return score[0] if score.size == 1 else score

    def classify(self, x):
        """Perform classification on samples in x.

        If a normalizer has been specified,
        input is normalized before classification.

        Classification is performed once wrt class 1, as
        the scores for class 0 just have inverted sign.

        Parameters
        ----------
        x : CArray or array_like
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).

        Returns
        -------
        y : CArray or scalar
            Flat dense array of shape (n_patterns,) with label assigned
            to each test pattern or a single scalar if n_patterns == 1.
        score : CArray
            Array of shape (n_patterns, 2) with classification
            score of each test pattern with respect to both classes.

        """
        # Discriminant function is called once (2 classes)
        s_tmp = CArray(
            self.discriminant_function(CArray(x).atleast_2d(), label=1))
        # Assembling scores for positive and negative class
        score = CArray([[-elem, elem] for elem in s_tmp])

        # Return a scalar if n_patterns == 1
        labels = CArray(score.argmax(axis=1)).ravel()
        labels = labels[0] if labels.size == 1 else labels

        return labels, score

    def _gradient_f(self, x=None, y=1):
        """Computes the gradient of the linear classifier's decision function
         wrt decision function input.

        For linear classifiers, the gradient wrt input is equal
        to the weights vector w. The point x can be in fact ignored.

        Parameters
        ----------
        x : CArray or None, optional
            The gradient is computed in the neighborhood of x.
        y : int, optional
            Binary index of the class wrt the gradient must be computed.
            Default is 1, corresponding to the positive class.

        Returns
        -------
        gradient : CArray
            The gradient of the linear classifier's decision function
            wrt decision function input. Vector-like array.

        """
        # Gradient sign depends on input label (0/1)
        return extend_binary_labels(y) * self.w
