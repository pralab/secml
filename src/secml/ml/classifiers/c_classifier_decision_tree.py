"""
.. module:: CClassifierDecisionTree
   :synopsis: Decision Tree classifier

.. moduleauthor:: Paolo Russu <paolo.russu@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import sklearn
from sklearn import tree

from secml.array import CArray
from secml.ml.classifiers import CClassifier


class CClassifierDecisionTree(CClassifier):
    """Decision Tree Classifier.

    Attributes
    ----------
    class_type : 'dec-tree'

    """
    __class_type = 'dec-tree'

    def __init__(self, criterion='gini', splitter='best',
                 max_depth=None, min_samples_split=2, preprocess=None):

        # Calling CClassifier constructor
        CClassifier.__init__(self, preprocess=preprocess)

        # Classifier Parameters
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self._dt = None

    def __clear(self):
        """Reset the object."""
        self._dt = None

    def __is_clear(self):
        """Returns True if object is clear."""
        return self._dt is None

    @property
    def min_samples_split(self):
        """Return decision tree min_samples_split."""
        return self._min_samples_split

    @min_samples_split.setter
    def min_samples_split(self, value):
        """Return decision tree min_samples_split."""
        self._min_samples_split = int(value)

    def _fit(self, dataset):
        """Trains the Decision Tree classifier."""
        if dataset.issparse is True and sklearn.__version__ < '0.16':
            raise ValueError(
                "sparse dataset is not supported if sklearn version < 0.16.")

        self._dt = tree.DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split)

        self._dt.fit(dataset.X.get_data(), dataset.Y.tondarray())

        return self._dt

    def _decision_function(self, x, y):
        """Computes the decision function (probability estimates) for each pattern in x.

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
        x = x.atleast_2d()  # Ensuring input is 2-D
        return CArray(self._dt.predict_proba(x.get_data())[:, y]).ravel()
