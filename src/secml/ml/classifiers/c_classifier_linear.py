"""
.. module:: CClassifierLinear
   :synopsis: Interface and common functions for linear classification

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from abc import ABCMeta

from secml.ml.classifiers import CClassifier
from secml.array import CArray
from secml.data import CDataset
from secml.utils.mixed_utils import check_is_fitted
from secml.core.decorators import deprecated


class CClassifierLinear(CClassifier, metaclass=ABCMeta):
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

    def _forward(self, x):
        """Computes the distance of each pattern in x to the hyperplane.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_samples,) if y is not None,
            otherwise a (n_samples, n_classes) array.

        """
        # Computing: `x * w^T`
        score = CArray(x.dot(self.w.T)).todense().ravel() + self.b

        scores = CArray.ones(shape=(x.shape[0], self.n_classes))
        scores[:, 0] = -score.ravel().T
        scores[:, 1] = score.ravel().T

        return scores

    def _backward(self, w):
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
        if w is not None:
            return w[0] * -self.w + w[1] * self.w
        else:
            raise ValueError("w cannot be set as None.")

    def grad_f_x(self, x, y=1):
        """Computes the gradient of the classifier's decision function wrt x.

        Parameters
        ----------
        x : CArray or None, optional
            The input point. The gradient will be computed at x.
        y : int
            Binary index of the class wrt the gradient must be computed.
            Default is y=1 to return gradient wrt the positive class.

        Returns
        -------
        gradient : CArray
            The gradient of the linear classifier's decision function
            wrt decision function input. Vector-like array.

        """
        return CClassifier.grad_f_x(self, x=x, y=y)
