"""
.. module:: ClassifierRandomForest
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

    def _discriminant_function(self, x, label):
        """Compute the discriminant function for pattern 'x'."""
        x_carray = CArray(x)
        if x_carray.issparse is True and sklearn.__version__ < '0.16':
            raise ValueError(
                "sparse input is not supported if sklearn version < 0.16.")

        return CArray(
            self._rf.predict_proba(x.get_data())[:, label]).ravel()
