"""
.. module:: CClassifierNearestCentroid
   :synopsis: Nearest Centroid Classifier

.. moduleauthor:: Biggio Battista <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn.neighbors import NearestCentroid

from secml.array import CArray
from secml.ml.classifiers import CClassifierSkLearn

from sklearn.metrics.pairwise import pairwise_distances


# TODO: EXPAND CLASS DOCSTRING
class CClassifierNearestCentroid(CClassifierSkLearn):
    """CClassifierNearestCentroid.

    Parameters
    ----------
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'nrst-centroid'

    """
    __class_type = 'nrst-centroid'

    def __init__(self, metric='euclidean',
                 shrink_threshold=None, preprocess=None):

        nc = NearestCentroid(metric=metric, shrink_threshold=shrink_threshold)

        super(CClassifierNearestCentroid, self).__init__(
            sklearn_model=nc, preprocess=preprocess)

    @property
    def metric(self):
        return CArray(self._sklearn_model.metric)

    @property
    def centroids(self):
        return CArray(self._sklearn_model.centroids_)

    def _decision_function(self, x, y=None):
        """ This sklearn classifier only supports predict.
        So we also implement a simple decision function
        based on pairwise distances.

        Parameters
        ----------
        x : CArray
            Input sample(s) after preprocessing
        y : {0, 1, ..., K-1} or None
            Class label of the output decision function.
            None returns all outputs.

        Returns
        -------
        CArray
            Negative distance values to centroids
            (i.e., similarity w/ centroid).

        """
        dist = CArray(pairwise_distances(
            x.get_data(), self._sklearn_model.centroids_,
            metric=self._sklearn_model.metric)).atleast_2d()

        return -dist if y is None else -dist[:, y].ravel()
