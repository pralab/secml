"""
.. module:: CClassifierLinear
   :synopsis: Interface and common functions for linear classification

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from abc import ABCMeta
import six

from secml.ml.classifiers import CClassifier
from secml.array import CArray
from secml.data import CDataset
from secml.utils.mixed_utils import check_is_fitted


@six.add_metaclass(ABCMeta)
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

    def _decision_function(self, x, y=None):
        """Computes the distance of each pattern in x to the hyperplane.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : {0, 1, None}
            The label of the class wrt the function should be calculated.
            If None, return the output for all classes.

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_samples,) if y is not None,
            otherwise a (n_samples, n_classes) array.

        """
        if y not in (0, 1, None):
            raise ValueError("decision function cannot be computed "
                             "against class {:}.".format(y))

        # Computing: `x * w^T`
        score = CArray(x.dot(self.w.T)).todense().ravel() + self.b

        scores = CArray.ones(shape=(x.shape[0], self.n_classes))
        scores[:, 0] = -score.ravel().T
        scores[:, 1] = score.ravel().T

        return scores[:, y].ravel() if y is not None else scores
