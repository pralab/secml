"""
.. module:: KNeighborsClassifier
   :synopsis: K-Neighbors classifier

.. moduleauthor:: Paolo Russu <paolo.russu@diee.unica.it>

"""

from sklearn import neighbors
import numpy as np
from secml.array import CArray
from secml.ml.classifiers import CClassifier


class CClassifierKNN(CClassifier):
    """K Neighbors Classifiers.

    Attributes
    ----------
    class_type : 'knn'

    """
    __class_type = 'knn'

    def __init__(self, n_neighbors=5, weights='uniform',
                 algorithm='auto', leaf_size=30, p=2,
                 metric='minkowski', metric_params=None,
                 preprocess=None):

        # Calling constructor of CClassifier
        CClassifier.__init__(self, preprocess=preprocess)

        self._n_neighbors = n_neighbors
        self._weights = weights
        self._algorithm = algorithm
        self.leaf_size = leaf_size
        self._p = p
        self._metric = metric
        self._metric_params = metric_params

        self._n_samples_training = 0
        self._tr_dataset = None
        self._KNC = None

    def __clear(self):
        """Reset the object."""
        self._n_samples_training = 0
        self._tr_dataset = None
        self._KNC = None

    def __is_clear(self):
        """Returns True if object is clear."""
        if self._n_samples_training != 0:
            return False
        if self._tr_dataset is not None:
            return False
        if self._KNC is not None:
            return False
        return True

    @property
    def n_neighbors(self):
        """Returns number of neighbors considered by the classifier."""
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, value):
        """Sets classifier n_neighbors."""
        self._n_neighbors = value

    @property
    def weights(self):
        """Returns weight function used in prediction."""
        return self._weights

    @weights.setter
    def weights(self, value):
        """Sets classifier weights."""
        self._weights = value

    @property
    def algorithm(self):
        """Returns classifier algorithm."""
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        """Sets classifier algorithm."""
        self._algorithm = value

    @property
    def leaf_size(self):
        """Returns classifier leaf_size."""
        return self._leaf_size

    @leaf_size.setter
    def leaf_size(self, value):
        """Sets classifier leaf_size."""
        self._leaf_size = value

    @property
    def p(self):
        """Returns type of distance used to group the samples."""
        return self._p

    @p.setter
    def p(self, value):
        """Sets classifier distance type."""
        self._p = value

    @property
    def metric(self):
        """Returns classifier metric."""
        return self._metric

    @metric.setter
    def metric(self, value):
        """Sets classifier metric."""
        self._metric = value

    @property
    def metric_params(self):
        """Returns classifier metric_params."""
        return self._metric_params

    @metric_params.setter
    def metric_params(self, value):
        """Sets classifier metric_params."""
        self._metric_params = value

    def _fit(self, dataset):
        """Trains the KNeighbors classifier.

        Training dataset is stored to use in kneighbors() method.

        """
        self._tr_dataset = dataset
        self._n_samples_training = dataset.Y.shape[0]

        self._KNC = neighbors.KNeighborsClassifier(self._n_neighbors,
                                                   self._weights,
                                                   self._algorithm,
                                                   self._leaf_size,
                                                   self._p,
                                                   self._metric,
                                                   self._metric_params)

        self._KNC.fit(dataset.X.get_data(), dataset.Y.tondarray())

        return self._KNC

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
        return CArray(self._KNC.predict_proba(x.get_data())[:, y]).ravel()

    def kneighbors(self, x, num_samples, return_distance=True):
        '''
        Find the training samples nearest to x
        
         Parameters
        ----------
        x : CArray
            The query point or points. 
        num_samples: int
            Number of neighbors to get
        return_distance: Bool
            If False, distances will not be returned

        Returns
        -------
        dist : CArray
            Array representing the lengths to points, only present if return_distance=True
        index_point: CArray
            Indices of the nearest points in the training set
        tr_dataset.X: CArray
            Training samples 


        '''
        dist, index_point = self._KNC.kneighbors(x.get_data(),
                                                 num_samples,
                                                 return_distance)

        index_point = CArray(index_point, dtype=np.int).ravel()

        return CArray(dist), index_point, self._tr_dataset.X[index_point, :]
