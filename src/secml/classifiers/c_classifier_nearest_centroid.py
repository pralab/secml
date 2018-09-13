"""
cambiare con _params 

.. module:: CClassifierNearestCentroid
   :synopsis: Nearest Centroid Classifier

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from prlib.array import CArray
from prlib.classifiers import CClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances


# TODO: EXPAND CLASS DOCSTRING
class CClassifierNearestCentroid(CClassifier):
    """CClassifierNearestCentroid."""
    class_type = 'nrst_centroid'

    def __init__(self, metric='euclidean',
                 shrink_threshold=None, normalizer=None):

        # Calling CClassifier init
        super(CClassifierNearestCentroid, self).__init__(normalizer=normalizer)

        self._metric = metric
        self._shrink_threshold = shrink_threshold

        self._nc = None
        self._centroids = None

    def __clear(self):
        """Reset the object."""
        self._nc = None
        self._centroids = None

    def is_clear(self):
        """Returns True if object is clear."""
        return self._nc is None and self._centroids is None and \
            super(CClassifierNearestCentroid, self).is_clear()

    @property
    def metric(self):
        return self._metric

    @property
    def centroids(self):
        return self._centroids

    def _train(self, dataset):
        """Trains classifier 
    
        Parameters
        ----------
        dataset : CDataset
            Binary (2-class) training set. Must be a :class:`.CDataset`
            instance with patterns data and corresponding labels.

        Returns
        -------
        trained_cls : CClassifierKernelDensityEstimator
            Instance of the KDE classifier trained using input dataset.

        """
        if dataset.num_classes > 2:
            raise ValueError("training can be performed on (1-classes) or "
                             "binary datasets only. If dataset is binary only "
                             "negative class are considered.")

        self._nc = NearestCentroid(self._metric, self._shrink_threshold)

        self._nc.fit(dataset.X.get_data(), dataset.Y.tondarray())

        self._centroids = CArray(self._nc.centroids_)

        return self._nc

    def _discriminant_function(self, x, label):
        """Compute the probability of samples to being in class with specified label.

        Score is the distance from the pattern with label x centroid 

        Parameters
        ----------
        x : CArray or array_like
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        label : int
            The label of the class with respect to which the function
            should be calculated.

        Returns
        -------
        score : CArray or scalar
            Probability for patterns to be in class with specified label

        """
        dist_from_ben_centroid = pairwise_distances(
            x.atleast_2d().get_data(),
            self.centroids[0, :].atleast_2d().get_data(), metric=self.metric)
        dis_from_mal_centroid = pairwise_distances(
            x.atleast_2d().get_data(),
            self.centroids[1, :].atleast_2d().get_data(), metric=self.metric)

        if label == 1:
            return CArray(
                dist_from_ben_centroid - dis_from_mal_centroid).ravel()
        else:
            return CArray(
                dis_from_mal_centroid - dist_from_ben_centroid).ravel()
