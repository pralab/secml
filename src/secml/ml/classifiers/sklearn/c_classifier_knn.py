"""
.. module:: KNeighborsClassifier
   :synopsis: K-Neighbors classifier

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
"""
from sklearn import neighbors

from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers import CClassifierSkLearn


class CClassifierKNN(CClassifierSkLearn):
    """K Neighbors Classifiers.

    Parameters
    ----------
    n_neighbors : int, optional
        Number of neighbors to use by default for :meth:`kneighbors` queries.
        Default 5.
    weights : str or callable, optional
        Weight function used in prediction. If 'uniform' (default), all points
        in each neighborhood are weighted equally; if 'distance' points are
        weighted by the inverse of their distance. Can also be an user-defined
        function which accepts an array of distances, and returns an array of
        the same shape containing the weights.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors. If 'auto' (default),
        the most appropriate algorithm is decided based on the values passed
        to :meth:`fit` method.
    leaf_size : int, optional
        Leaf size passed to BallTree or KDTree. Default 30.
    p : int, optional
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2 (default). For arbitrary p, minkowski_distance (l_p)
        is used.
    metric : str or callable, optional
        The distance metric to use for the tree. If 'minkowski' (default) and
        p = 2, it is equivalent to the standard Euclidean metric.
        If metric is 'precomputed', X is assumed to be a distance matrix and
        must be square during fit.
    metric_params : dict, optional
        Additional keyword arguments for the metric function.
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'knn'

    """
    __class_type = 'knn'

    def __init__(self, n_neighbors=5, weights='uniform',
                 algorithm='auto', leaf_size=30, p=2,
                 metric='minkowski', metric_params=None,
                 preprocess=None):

        self._tr = None

        knn = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p,
            leaf_size=leaf_size, metric=metric, metric_params=metric_params)

        CClassifierSkLearn.__init__(self, sklearn_model=knn,
                                    preprocess=preprocess)

    @property
    def tr(self):
        """Training set."""
        return self._tr

    def _fit(self, x, y):
        """Trains the KNeighbors classifier.

        Training dataset is stored to use in kneighbors() method.

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features)
        y : CArray
            Array of shape (n_samples,) containing the class labels.

        Returns
        -------
        CClassifierKNN
            Trained classifier.

        """
        self._tr = CDataset(x, y)
        return CClassifierSkLearn._fit(self, x, y)

    def kneighbors(self, x, num_samples=None):
        """
        Find the training samples nearest to x
        
         Parameters
        ----------
        x : CArray
            The query point or points. 
        num_samples: int or None
            Number of neighbors to get. if None, use n_neighbors

        Returns
        -------
        dist : CArray
            Array representing the lengths to points
        index_point: CArray
            Indices of the nearest points in the training set
        tr_dataset.X: CArray
            Training samples
        """
        if num_samples is None:
            num_samples = self._sklearn_model.n_neighbors

        dist, index_point = self._sklearn_model.kneighbors(
            x.get_data(), num_samples, return_distance=True)

        index_point = CArray(index_point, dtype=int).ravel()

        return CArray(dist), index_point, self._tr.X[index_point, :]
