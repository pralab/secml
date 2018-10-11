"""
.. module:: CClassifierRandomForest
   :synopsis: Random Forest classifier

.. moduleauthor:: Paolo Russu <paolo.russu@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import sklearn
from sklearn import ensemble
import warnings

from secml.array import CArray
from secml.classifiers import CClassifier


class CClassifierRandomForest(CClassifier):
    """Random Forest classifier."""
    class_type = 'random_forest'

    def __init__(self, n_estimators=10, criterion='gini',
                 max_depth=None, min_samples_split=2, normalizer=None):

        # Calling CClassifier constructor
        CClassifier.__init__(self, normalizer=normalizer)

        # Classifier Parameters
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self._rf = None

    def __clear(self):
        """Reset the object."""
        self._rf = None

    def is_clear(self):
        """Returns True if object is clear."""
        return self._rf is None and \
            super(CClassifierRandomForest, self).is_clear()

    @property
    def n_estimators(self):
        """Returns classifier estimators."""
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, value):
        """Sets classifier estimators."""
        self._n_estimators = int(value)

    @property
    def min_samples_split(self):
        """Returns classifier min_samples_split."""
        return self._min_samples_split

    @min_samples_split.setter
    def min_samples_split(self, value):
        """Sets classifier min_samples_split."""
        self._min_samples_split = value

    def _train(self, dataset):
        """Trains the Random Forest classifier."""
        if dataset.issparse is True and sklearn.__version__ < '0.16':
            raise ValueError(
                "sparse dataset is not supported if sklearn version < 0.16.")

        self._rf = ensemble.RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split)

        self._rf.fit(dataset.X.get_data(), dataset.Y.tondarray())

        return self._rf

    def _discriminant_function(self, x, y):
        """Computes the discriminant function (probability estimates) for each pattern in x.

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
            Value of the discriminant function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        x = x.atleast_2d()  # Ensuring input is 2-D
        return CArray(self._rf.predict_proba(x.get_data())[:, y]).ravel()
