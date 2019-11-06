"""
.. module:: CDataLoaderSklearn
   :synopsis: Collection of dataset loaders from sklearn library.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from multiprocessing import Lock

from abc import ABCMeta, abstractmethod

from secml.data.loader import CDataLoader
from secml.data import CDataset
from secml.array import CArray

__all__ = ['CDLRandom', 'CDLRandomRegression',
           'CDLRandomBlobs', 'CDLRandomBlobsRegression',
           'CDLRandomCircles', 'CDLRandomCircleRegression',
           'CDLRandomMoons', 'CDLRandomBinary',
           'CDLIris', 'CDLDigits', 'CDLBoston', 'CDLDiabetes']


class CDLRandom(CDataLoader):
    """Class for loading random data.

    Generate a random n-class classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of a 2 * class_sep-sided hypercube, and assigns an equal
    number of clusters to each class.

    It introduces interdependence between these features and adds various
    types of further noise to the data.

    Prior to shuffling, X stacks a number of these primary "informative"
    features, "redundant" linear combinations of these,
    "repeated" duplicates of sampled features,
    and arbitrary noise for and remaining features.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.
    n_features : int, optional (default=20)
        The total number of features.
        These comprise n_informative informative features,
        n_redundant redundant features, n_repeated duplicated
        features and ``n_features - n_informative
        - n_redundant - n_repeated``
        useless features drawn at random.
    n_informative : int, optional (default=2)
        The number of informative features.
        Each class is composed of a number of gaussian clusters each
        located around the vertices of a hypercube in a subspace of
        dimension n_informative. For each cluster, informative
        features are drawn independently from N(0, 1) and then randomly
        linearly combined within each cluster in order to add covariance.
        The clusters are then placed on the vertices of the hypercube.
    n_redundant : int, optional (default=2)
        The number of redundant features.
        These features are generated as random linear combinations of
        the informative features.
    n_repeated : int, optional (default=0)
        The number of duplicated features, drawn randomly from the
        informative and the redundant features.
    n_classes : int, optional (default=2)
        The number of classes (or labels) of the classification problem.
    n_clusters_per_class : int, optional (default=2)
        The number of clusters per class.
    weights : list of floats or None (default=None)
        The proportions of samples assigned to each class.
        If None, then classes are balanced. Note that if
        ``len(weights) == n_classes - 1``, then the last
        class weight is automatically inferred.
        More than n_samples samples may be returned if the sum
        of weights exceeds 1.
    flip_y : float, optional (default=0.01)
        The fraction of samples whose class are randomly exchanged.
    class_sep : float, optional (default=1.0)
        The factor multiplying the hypercube dimension.
    hypercube : bool, optional (default=True)
        If True, the clusters are put on the vertices of a hypercube.
        If False, the clusters are put on the vertices
        of a random polytope.
    shift : float, array of shape [n_features] or None, optional (default=0.0)
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].
    scale : float, array of shape [n_features] or None, optional (default=1.0)
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, is the RandomState instance used by np.random.

    Attributes
    ----------
    class_type : 'classification'

    """
    __class_type = 'classification'

    def __init__(self, n_samples=100, n_features=20, n_informative=2,
                 n_redundant=2, n_repeated=0, n_classes=2,
                 n_clusters_per_class=2, weights=None,
                 flip_y=0.01, class_sep=1.0, hypercube=True,
                 shift=0.0, scale=1.0, random_state=None):

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.n_repeated = n_repeated
        self.n_classes = n_classes
        self.n_clusters_per_class = n_clusters_per_class
        self.weights = weights
        self.flip_y = flip_y
        self.class_sep = class_sep
        self.hypercube = hypercube
        self.shift = shift
        self.scale = scale
        self.random_state = random_state

    def load(self):
        """Loads the dataset.

        Returns
        -------
        dataset : CDataset
            The randomly generated dataset.

        """
        from sklearn.datasets import make_classification
        patterns, labels = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            n_repeated=self.n_repeated,
            n_classes=self.n_classes,
            n_clusters_per_class=self.n_clusters_per_class,
            weights=self.weights,
            flip_y=self.flip_y,
            class_sep=self.class_sep,
            hypercube=self.hypercube,
            shift=self.shift,
            scale=self.scale,
            random_state=self.random_state)
        return CDataset(patterns, labels)


class CDLRandomRegression(CDataLoader):
    """Generate a random regression problem.

    The input set can either be well conditioned (by default) or have a low
    rank-fat tail singular profile.

    The output is generated by applying a (potentially biased)
    random linear regression model with `n_informative` nonzero
    regressors to the previously generated input and some gaussian
    centered noise with some adjustable scale.
    
    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.
    n_features : int, optional (default=100)
        The number of features.
    n_informative : int, optional (default=10)
        The number of informative features, i.e.,
        the number of features used to build the linear model
        used to generate the output.
    n_targets : int, optional (default=1)
        The number of regression targets, i.e.,
        the dimension of the y output vector associated with a sample.
        By default, the output is a scalar.
    bias : float, optional (default=0.0)
        The bias term in the underlying linear model.
    effective_rank : int or None, optional (default=None)
        if not None:
            The approximate number of singular vectors
            required to explain most of the input data
            by linear combinations.
            Using this kind ofsingular spectrum in the input
            allows the generator to reproduce
            the correlations often observed in practice.
        if None:
            The input set is well conditioned, centered and gaussian with
            unit variance.
    tail_strength : float between 0.0 and 1.0, optional (default=0.5)
        The relative importance of the fat noisy
        tail of the singular values
        profile if `effective_rank` is not None.
    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise applied to the output.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, is the RandomState instance used by np.random.

    Attributes
    ----------
    class_type : 'regression'

    """
    __class_type = 'regression'

    def __init__(self, n_samples=100, n_features=100, n_informative=10,
                 n_targets=1, bias=0.0, effective_rank=None,
                 tail_strength=0.5, noise=0.0, random_state=None):

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_targets = n_targets
        self.bias = bias
        self.effective_rank = effective_rank
        self.tail_strength = tail_strength
        self.noise = noise
        self.random_state = random_state

    def load(self):
        """Loads the dataset.

        Returns
        -------
        dataset : CDataset
            The randomly generated dataset.

        """
        from sklearn.datasets import make_regression
        patterns, labels = make_regression(n_samples=self.n_samples,
                                           n_features=self.n_features,
                                           n_informative=self.n_informative,
                                           n_targets=self.n_targets,
                                           bias=self.bias,
                                           effective_rank=self.effective_rank,
                                           tail_strength=self.tail_strength,
                                           noise=self.noise,
                                           random_state=self.random_state)
        return CDataset(patterns, labels)


class CDLRandomBlobs(CDataLoader):
    """Generate isotropic Gaussian blobs for clustering.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points equally divided among clusters.
    n_features : int, optional (default=2)
        The number of features for each sample.
        This parameter will not be considered if centers is different
        from None
    centers : int or array of shape [n_centers, n_features]
        The number of centers to generate (default=3),
        or the fixed center locations as list of tuples
    cluster_std: float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.
    center_box : pair of floats (min, max), optional (default=(-10.0, 10.0))
        The bounding box for each cluster center
        when centers are generated at random.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, is the RandomState instance used by np.random.

    Attributes
    ----------
    class_type : 'blobs'

    """
    __class_type = 'blobs'

    def __init__(self, n_samples=100, n_features=2, centers=3,
                 cluster_std=1.0, center_box=(-10.0, 10.0), random_state=None):

        self.n_samples = n_samples
        self.n_features = n_features
        self.cluster_std = cluster_std
        self.centers = centers
        self.center_box = center_box
        self.random_state = random_state

    def load(self):
        """Loads the dataset.

        Returns
        -------
        dataset : CDataset
            The randomly generated dataset.

        """
        from sklearn.datasets import make_blobs
        patterns, labels = make_blobs(
            n_samples=self.n_samples,
            n_features=self.n_features,
            centers=self.centers,
            cluster_std=self.cluster_std,
            center_box=self.center_box,
            random_state=self.random_state)

        return CDataset(patterns, labels)


class CDLRandomBlobsRegression(CDataLoader):
    """This class loads blobs regression.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points equally divided among clusters.
    centers : int or array of shape [n_centers, n_features], optional (default=3)
        The number of centers to generate, or the fixed center locations.
    cluster_std: list of floats, optional (default=(1.0,1.0))
        The standard deviation of the clusters.
    bias : bias that will sum to the function
    w : the height of every gaussian
    centers: list of tuple optional (default=([1,1],[-1,-1]))
        The bounding box for each cluster center when centers are
        generated at random.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, is the RandomState instance used by np.random.

    Attributes
    ----------
    class_type : 'blobs-regression'

    """
    __class_type = 'blobs-regression'

    def __init__(self, n_samples=100, cluster_std=(1.0, 1.0),
                 bias=1.0, w=(2.0, -1.0), centers=([0, 0], [-1, -1]),
                 random_state=None):

        self.n_samples = n_samples
        self.bias = bias
        self.w = w
        self.centers = centers
        self.cluster_std = cluster_std
        self.random_state = random_state

    def _dts_function(self, X):
        """ TODO: Put a comment for this function. """
        from secml.ml.stats import CDistributionGaussian
        d = X.shape[1]  # number of features
        Y = self.bias
        for gauss_idx in range(len(self.centers)):
            Y += self.w[gauss_idx] * \
                 CDistributionGaussian(mean=self.centers[gauss_idx],
                                       cov=self.cluster_std[gauss_idx] *
                                       CArray.eye(d, d)).pdf(X)
        return Y

    def load(self):
        """Loads the dataset.

        Returns
        -------
        dataset : CDataset
            The randomly generated dataset.

        """
        from sklearn.datasets import make_blobs
        patterns = make_blobs(
            n_samples=self.n_samples, n_features=2, centers=self.centers,
            cluster_std=self.cluster_std, random_state=self.random_state)[0]
        return CDataset(patterns, self._dts_function(CArray(patterns)))


class CDLRandomCircles(CDataLoader):
    """Make a large circle containing a smaller circle in 2d.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    factor : double < 1 (default=.8)
        Scale factor between inner and outer circle.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, is the RandomState instance used by np.random.

    Attributes
    ----------
    class_type : 'circles'

    """
    __class_type = 'circles'

    def __init__(self, n_samples=100, noise=None,
                 factor=0.8, random_state=None):

        self.n_samples = n_samples
        self.noise = noise
        self.factor = factor
        self.random_state = random_state

    def load(self):
        """Loads the dataset.

        Returns
        -------
        dataset : CDataset
            The randomly generated dataset.

        """
        from sklearn.datasets import make_circles
        patterns, labels = make_circles(
            n_samples=self.n_samples,
            noise=self.noise,
            factor=self.factor,
            random_state=self.random_state)
        return CDataset(patterns, labels)


class CDLRandomCircleRegression(CDataLoader):
    """Make a large circle containing a smaller circle in 2d.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    factor : double < 1 (default=.8)
        Scale factor between inner and outer circle.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, is the RandomState instance used by np.random.

    Attributes
    ----------
    class_type : 'circles-regression'

    """
    __class_type = 'circles-regression'

    def __init__(self, n_samples=100, noise=None,
                 factor=0.8, random_state=None):

        self.n_samples = n_samples
        self.noise = noise
        self.factor = factor
        self.random_state = random_state

    def _dts_function(self, X):
        """TODO: Add comment for this function!"""
        return X[:, 0] ** 2 + X[:, 1] ** 2

    def load(self):
        """Loads the dataset.

        Returns
        -------
        dataset : CDataset
            The randomly generated dataset.

        """
        from sklearn.datasets import make_circles
        patterns = make_circles(
            n_samples=self.n_samples,
            noise=self.noise,
            factor=self.factor,
            random_state=self.random_state)[0]
        return CDataset(patterns, self._dts_function(patterns))


class CDLRandomMoons(CDataLoader):
    """Make two interleaving half circles.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, is the RandomState instance used by np.random.

    Attributes
    ----------
    class_type : 'moons'

    """
    __class_type = 'moons'

    def __init__(self, n_samples=100, noise=None, random_state=None):

        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state

    def load(self):
        """Loads the dataset.

        Returns
        -------
        dataset : CDataset
            The randomly generated dataset.

        """
        from sklearn.datasets import make_moons
        patterns, labels = make_moons(
            n_samples=self.n_samples,
            noise=self.noise,
            random_state=self.random_state)
        return CDataset(patterns, labels)


class CDLRandomBinary(CDataLoader):
    """Generate random binary data.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.
    n_features : int, optional (default=2)
        The total number of features

    Attributes
    ----------
    class_type : 'binary'

    """
    __class_type = 'binary'

    def __init__(self, n_samples=100, n_features=2):

        self.n_samples = n_samples
        self.n_features = n_features

    def load(self):
        """Loads the dataset.

        Returns
        -------
        dataset : CDataset
            The randomly generated dataset.

        """
        patterns = CArray.randint(2, shape=(self.n_samples, self.n_features))
        labels = CArray.randint(2, shape=(1, self.n_samples))
        return CDataset(patterns, labels)


class CDLRandomToy(CDataLoader, metaclass=ABCMeta):
    """Loads a random toy dataset (abstract interface).

    Available toy datasets:
     - iris (classification) -> `CDLIris`
     - digits (classification) -> `CDLDigits`
     - boston (regression) -> `CDLBoston`
     - diabetes (regression) -> `CDLDiabetes`

    Parameters
    ----------
    class_list : list of string (default None)
        Each string is the name of data's class that we want
        in the new dataset.  If None every class will be keep
    zero_one : bool
        If is true, and class list is equal to two, will be
        assigned 0 at the label with lower value, 1 to the other.

    """
    __lock = Lock()  # Lock to prevent multiple parallel download/extraction

    def __init__(self, class_list=None, zero_one=False):

        self.class_list = class_list
        self.zero_one = zero_one

    @property
    @abstractmethod
    def toy(self):
        """Identifier of the toy dataset."""
        raise NotImplementedError

    def _select_classes(self, class_list, patterns, labels):

        sel_patterns = None
        sel_labels = None

        for single_class in class_list:
            this_class_pat_idx = labels.find(labels == single_class)

            if sel_patterns is None:
                sel_patterns = patterns[this_class_pat_idx, :]
                sel_labels = labels[this_class_pat_idx]
            else:
                sel_patterns = sel_patterns.append(
                    patterns[this_class_pat_idx, :], axis=0)
                sel_labels = sel_labels.append(
                    labels[this_class_pat_idx])

        if self.zero_one is True:
            if len(class_list) > 2:
                raise ValueError("you are try to convert to 0 1 label for a "
                                 "dataset with more than 2 classes")
            else:
                class_list.sort()
                sel_labels[sel_labels == class_list[0]] = 0
                sel_labels[sel_labels == class_list[1]] = 1

        return CDataset(sel_patterns, sel_labels)

    def load(self):
        """Loads the dataset.

        Returns
        -------
        dataset : CDataset
            The randomly generated dataset.

        """
        with CDLRandomToy.__lock:
            if self.toy == 'iris':
                from sklearn.datasets import load_iris
                toy_data = load_iris()
            elif self.toy == 'digits':
                from sklearn.datasets import load_digits
                toy_data = load_digits()
            elif self.toy == 'boston':
                from sklearn.datasets import load_boston
                toy_data = load_boston()
            elif self.toy == 'diabetes':
                from sklearn.datasets import load_diabetes
                toy_data = load_diabetes()
            else:
                raise ValueError("toy dataset {:} if not available.".format(self.toy))

        # Returning a CDataset
        if self.class_list is None:
            return CDataset(CArray(toy_data.data), CArray(toy_data.target))
        else:
            return self._select_classes(self.class_list,
                                        CArray(toy_data.data),
                                        CArray(toy_data.target))


class CDLIris(CDLRandomToy):
    """Loads Iris dataset.

    The iris dataset is a classic and very easy multi-class
    classification dataset.

    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============

    Parameters
    ----------
    class_list : list of str (default None)
        Each string is the name of data's class that we want
        in the new dataset.  If None every class will be keep
    zero_one : bool
        If is true, and class list is equal to two, will be
        assigned 0 at the label with lower value, 1 to the other.

    Attributes
    ----------
    class_type : 'iris'

    """
    __class_type = 'iris'
    toy = 'iris'


class CDLDigits(CDLRandomToy):
    """Loads Digits dataset.

    The digits dataset is a classic and very easy multi-class
    classification dataset. Each datapoint is a 8x8 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class             ~180
    Samples total                 1797
    Dimensionality                  64
    Features             integers 0-16
    =================   ==============

    Parameters
    ----------
    class_list : list of str (default None)
        Each string is the name of data's class that we want
        in the new dataset.  If None every class will be keep
    zero_one : bool
        If is true, and class list is equal to two, will be
        assigned 0 at the label with lower value, 1 to the other.

    Attributes
    ----------
    class_type : 'digits'

    """
    __class_type = 'digits'
    toy = 'digits'


class CDLBoston(CDLRandomToy):
    """Loads Boston dataset.

    Boston house-prices dataset, useful for regression.

    ==============     ==============
    Samples total                 506
    Dimensionality                 13
    Features           real, positive
    Targets             real 5. - 50.
    ==============     ==============

    Parameters
    ----------
    class_list : list of str (default None)
        Each string is the name of data's class that we want
        in the new dataset.  If None every class will be keep
    zero_one : bool
        If is true, and class list is equal to two, will be
        assigned 0 at the label with lower value, 1 to the other.

    Attributes
    ----------
    class_type : 'boston'

    """
    __class_type = 'boston'
    toy = 'boston'


class CDLDiabetes(CDLRandomToy):
    """Loads Diabetes dataset.

    Diabetes dataset, useful for regression.

    ==============      ==================
    Samples total       442
    Dimensionality      10
    Features            real, -.2 < x < .2
    Targets             integer 25 - 346
    ==============      ==================

    Parameters
    ----------
    class_list : list of str (default None)
        Each string is the name of data's class that we want
        in the new dataset.  If None every class will be keep
    zero_one : bool
        If is true, and class list is equal to two, will be
        assigned 0 at the label with lower value, 1 to the other.

    Attributes
    ----------
    class_type : 'diabetes'

    """
    __class_type = 'diabetes'
    toy = 'diabetes'
