"""
.. module:: CClassifierNearestCentroid
   :synopsis: Nearest Centroid Classifier

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances

from secml.array import CArray
from secml.ml.classifiers import CClassifierInterface
from secml.ml.classifiers.clf_utils import convert_binary_labels
from secml.utils.mixed_utils import check_is_fitted


# TODO: EXPAND CLASS DOCSTRING
class CClassifierNearestCentroid(CClassifierInterface):
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

        # Calling CClassifier init
        super(CClassifierNearestCentroid, self).__init__(preprocess=preprocess)

        self._metric = metric
        self._shrink_threshold = shrink_threshold

        self._nc = None
        self._centroids = None

    @property
    def metric(self):
        return self._metric

    @property
    def centroids(self):
        return self._centroids

    def _check_is_fitted(self):
        """Check if the classifier is trained (fitted).

        Raises
        ------
        NotFittedError
            If the classifier is not fitted.

        """
        check_is_fitted(self, 'centroids')
        super(CClassifierNearestCentroid, self)._check_is_fitted()

    def _fit(self, dataset):
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

    def decision_function(self, x, y=1):
        """Computes the decision function for each pattern in x.

        The score is the distance of each pattern
         from the centroid of class `y`

        If a preprocess has been specified, input is normalized
         before computing the decision function.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : {0, 1}, optional
            The label of the class wrt the function should be calculated.
            Default is 1.

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        self._check_is_fitted()

        x = x.atleast_2d()  # Ensuring input is 2-D

        # Transform data if a preprocess is defined
        x = self._preprocess_data(x)

        sign = convert_binary_labels(y)  # Sign depends on input label (0/1)

        return sign * self._decision_function(x)

    def _decision_function(self, x, y=1):
        """Computes the decision function for each pattern in x.

        The score is the distance of each pattern
         from the centroid of class `label`

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : {0,1}
            The label of the class wrt the function should be calculated.

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        x = x.atleast_2d()  # Ensuring input is 2-D

        dist_from_ben_centroid = pairwise_distances(
            x.get_data(), self.centroids[0, :].atleast_2d().get_data(),
            metric=self.metric)
        dis_from_mal_centroid = pairwise_distances(
            x.get_data(), self.centroids[1, :].atleast_2d().get_data(),
            metric=self.metric)

        score = CArray(dist_from_ben_centroid - dis_from_mal_centroid).ravel()
        sign = convert_binary_labels(y)  # adjust sign based on y
        return sign * score
