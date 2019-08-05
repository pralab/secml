"""
.. module:: CClassifierMCSLinear
   :synopsis: Multiple Linear Classifier System

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from __future__ import division
from six.moves import range

from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers import CClassifierLinear
from secml.ml.classifiers.gradients import CClassifierGradientLinearMixin


class CClassifierMCSLinear(CClassifierLinear, CClassifierGradientLinearMixin):
    """MCS averaging a set of LINEAR classifiers.

    Eventually, one yields a linear classifier itself,
    where w (b) is the average of the feature weights (bias)
    of the base classifiers.

    Parameters
    ----------
    classifier : CClassifierLinear
        Instance of the linear classifier to be used in the MCS.
    num_classifiers : int, optional
        Number of linear classifiers to fit, default 10.
    max_samples : float, optional
        Percentage of the samples to use for training,
        range [0, 1.0]. Default 1.0 (all the samples).
    max_features : float, optional
        Percentage of the features to use for training,
        range [0, 1.0]. Default 1.0 (all the features.
    random_state : int or None, optional
        If int, random_state is the seed used by the random number generator.
        If None, no fixed seed will be set.
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'mcs-linear'

    """
    __class_type = 'mcs-linear'
    
    def __init__(self, classifier, num_classifiers=10,
                 max_samples=1.0, max_features=1.0,
                 random_state=None, preprocess=None):

        # Calling constructor of CClassifierLinear
        CClassifierLinear.__init__(self, preprocess=preprocess)

        # Instance of the classifier to use
        self.classifier = classifier
        # Classifier parameters
        self.n_classifiers = num_classifiers
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state

    @property
    def classifier(self):
        """Instance of the linear classifier used in the MCS."""
        return self._classifier

    @classifier.setter
    def classifier(self, clf):
        # Binary classifier to use
        if not isinstance(clf, CClassifierLinear):
            raise TypeError("MCS classifier is only available "
                            "for linear classifiers.")
        self._classifier = clf
        
    @property
    def n_classifiers(self):
        """Number of linear classifiers to fit."""
        return self._n_classifiers
    
    @n_classifiers.setter
    def n_classifiers(self, value):
        self._n_classifiers = int(value)

    @property
    def max_samples(self):
        """Percentage of the samples to use for training."""
        return self._max_samples
    
    @max_samples.setter
    def max_samples(self, value):
        if 0 > value or value > 1:
            raise ValueError("`max_samples` must be inside [0, 1.0] range.")
        self._max_samples = float(value)

    @property
    def max_features(self):
        return self._max_features
    
    @max_features.setter
    def max_features(self, value):
        """Percentage of the features to use for training."""
        if 0 > value or value > 1:
            raise ValueError("`max_features` must be inside [0, 1.0] range.")
        self._max_features = float(value)

    def _fit(self, dataset):
        """Fit the MCS Linear Classifier.

        Parameters
        ----------
        dataset : CDataset
            Binary (2-classes) training set. Must be a :class:`.CDataset`
            instance with patterns data and corresponding labels.

        Returns
        -------
        trained_cls : CClassifierMCSLinear
            Instance of the MCS linear classifier trained using input dataset.

        """
        num_samples = int(self.max_samples * dataset.num_samples)
        num_features = int(self.max_features * dataset.num_features)

        self._w = CArray.zeros(dataset.num_features, sparse=dataset.issparse)
        self._b = CArray(0.0)
        
        for i in range(self.n_classifiers):

            # generate random indices for features and samples
            idx_samples = CArray.randsample(dataset.num_samples, num_samples,
                                            random_state=self.random_state)
            idx_features = CArray.randsample(dataset.num_features, num_features,
                                             random_state=self.random_state)

            data_x = dataset.X[idx_samples, :]
            data_x = data_x[:, idx_features]

            data = CDataset(data_x, dataset.Y[idx_samples])
            
            self.classifier.fit(data)
            self._w[idx_features] += self.classifier.w
            self._b += self.classifier.b
            
        self._w /= self.n_classifiers
        self._b /= self.n_classifiers
        self._b = self._b[0]  # The bias is a scalar

        return self
