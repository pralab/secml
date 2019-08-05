"""
.. module:: KNeighborsClassifier
   :synopsis: K-Neighbors classifier

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
"""
from sklearn import neighbors

from secml.array import CArray
from secml.ml.classifiers import CClassifierSkLearn


class CClassifierKNN(CClassifierSkLearn):
    """K Neighbors Classifiers.

    Parameters
    ----------
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

        self._tr_dataset = None

        knn = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p,
            leaf_size=leaf_size, metric=metric, metric_params=metric_params)

        CClassifierSkLearn.__init__(self, sklearn_model=knn,
                                    preprocess=preprocess)

    def _fit(self, dataset):
        """Trains the KNeighbors classifier.

        Training dataset is stored to use in kneighbors() method.

        """
        self._tr_dataset = dataset  # TODO: do we need this?
        return CClassifierSkLearn._fit(self, dataset)

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

        return CArray(dist), index_point, self._tr_dataset.X[index_point, :]
