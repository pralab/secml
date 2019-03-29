"""
.. module:: CClassifierLinear
   :synopsis: Interface and common functions for linear classification

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml.ml.classifiers import CClassifier
from secml.ml.classifiers.clf_utils import convert_binary_labels
from secml.array import CArray
from secml.data import CDataset
from secml import _NoValue
from secml.utils.mixed_utils import check_is_fitted


class CClassifierLinear(CClassifier):
    """Abstract class that defines basic methods for linear classifiers.

    A linear classifier assign a label (class) to new patterns
    computing the inner product between the patterns and a vector
    of weights for each training set feature.

    This interface implements a set of generic methods for training
    and classification that can be used for every linear model.

    Parameters
    ----------
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    """

    def __init__(self, preprocess=None):
        # Linear classifier parameters
        self._w = None
        self._b = None

        # Calling init of CClassifier
        CClassifier.__init__(self, preprocess=preprocess)

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
        if self.preprocess is None or \
                self.preprocess is not None and self.preprocess.is_linear():
            return True
        return False

    def _check_is_fitted(self):
        """Check if the classifier is trained (fitted).

        Raises
        ------
        NotFittedError
            If the classifier is not fitted.

        """
        # Do not check `b` as some classifiers do not set it
        check_is_fitted(self, 'w')
        super(CClassifierLinear, self)._check_is_fitted()

    def fit(self, dataset, n_jobs=1):
        """Trains the linear classifier.

        If a preprocess has been specified,
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

        return super(CClassifierLinear, self).fit(dataset, n_jobs=n_jobs)

    def _decision_function(self, x, y=1):
        """Computes the distance from the separating hyperplane for each pattern in x.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : {1}
            The label of the class wrt the function should be calculated.
            decision function is always computed wrt positive class (1).

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        if y != 1:
            raise ValueError(
                "decision function is always computed wrt positive class.")
        x = x.atleast_2d()  # Ensuring input is 2-D
        # Computing: `x * w^T`
        return CArray(x.dot(self.w.T)).todense().ravel() + self.b

    def decision_function(self, x, y=1):
        """Computes the decision function for each pattern in x.

        For a linear classifier the decision function is given by::

            .. math:: f[i] =  (x[i] * w^T) + b

        If a preprocess has been specified, input is normalized
        before computing the decision function.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : {0, 1}, optional
            The label of the class wrt the function should be calculated.
            Default is 1.

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

        sign = convert_binary_labels(y)  # Sign depends on input label (0/1)

        return sign * self._decision_function(x)

    def predict(self, x, return_decision_function=False, n_jobs=_NoValue):
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

        Returns
        -------
        labels : CArray
            Flat dense array of shape (n_patterns,) with the label assigned
             to each test pattern. The classification label is the label of
             the class associated with the highest score.
        scores : CArray, optional
            Array of shape (n_patterns, 1) with classification
             score of each test pattern with respect to {0, +1} classes.
            Will be returned only if `return_decision_function` is True.

        """
        if n_jobs is not _NoValue:
            raise ValueError("`n_jobs` not supported")

        # decision function is called once (2 classes)
        s_tmp = CArray(
            self.decision_function(CArray(x).atleast_2d(), y=1))
        # Assembling scores for positive and negative class
        scores = CArray([[-elem, elem] for elem in s_tmp])

        # The classification label is the label of the class
        # associated with the highest score
        labels = scores.argmax(axis=1).ravel()

        return (labels, scores) if return_decision_function is True else labels

    def gradient_f_x(self, x, y=1, **kwargs):
        """Computes the gradient of the classifier's output wrt input.

        Parameters
        ----------
        x : CArray
            The gradient is computed in the neighborhood of x.
        y : int, optional
            Index of the class wrt the gradient must be computed. Default 1.
        **kwargs
            Optional parameters for the function that computes the
            gradient of the decision function. See the description of
            each classifier for a complete list of optional parameters.

        Returns
        -------
        gradient : CArray
            Gradient of the classifier's output wrt input. Vector-like array.

        """
        return super(CClassifierLinear, self).gradient_f_x(x, y, **kwargs)

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
        return convert_binary_labels(y) * self.w
