"""
.. module:: ClassifierSGD
   :synopsis: Stochastic Gradient Descent (SGD) classifier

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Paolo Russu <paolo.russu@diee.unica.it>

"""
from sklearn import linear_model

from prlib.classifiers import CClassifierLinear
from prlib.array import CArray
from prlib.classifiers.regularizer import CRegularizer
from prlib.classifiers.loss import CLoss
from prlib.kernel import CKernel


class CClassifierSGD(CClassifierLinear):
    """Stochastic Gradient Descent Classifier."""
    class_type = 'sgd'

    def __init__(self, loss, regularizer, kernel=None, alpha=0.01,
                 n_iter=5, shuffle=True, class_weight=None,
                 fit_intercept=True, warm_start=False, normalizer=None,
                 learning_rate='optimal', eta0=10.0, power_t=0.5):

        # Calling the superclass init
        CClassifierLinear.__init__(self, normalizer=normalizer)

        # Keep private (not an sklearn sgd parameter)
        self._loss = CLoss.create(loss)
        # Keep private (not an sklearn sgd parameter)
        self._regularizer = CRegularizer.create(regularizer)

        # Classifier parameters
        self.alpha = alpha
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.class_weight = class_weight
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t

        # Similarity function (bound) to use for computing features
        # Keep private (not a param of SGD)
        self._kernel = kernel if kernel is None else CKernel.create(kernel)

        # After training attributes
        self._tr = None  # slot for the training data

    def __clear(self):
        """Reset the object."""
        self._tr = None

    def is_clear(self):
        """Returns True if object is clear."""
        return self._tr is None and super(CClassifierSGD, self).is_clear()

    @property
    def loss(self):
        """Returns the loss function used by classifier."""
        return self._loss

    @property
    def regularizer(self):
        """Returns the regularizer function used by classifier."""
        return self._regularizer

    @property
    def alpha(self):
        """Returns the Constant that multiplies the regularization term."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """Sets the Constant that multiplies the regularization term."""
        self._alpha = float(value)

    @property
    def class_weight(self):
        """Weight of each training class."""
        return self._class_weight

    @class_weight.setter
    def class_weight(self, value):
        """Sets the weight of each training class."""
        if isinstance(value, dict) and len(value) != 2:
            raise ValueError("weight of positive (+1) and negative (0) "
                             "classes only must be specified.")
        self._class_weight = value

    @property
    def eta0(self):
        """The initial learning rate for the `invscaling` learning rate.
        Default is 10.0 (corresponding to sqrt(1.0/sqrt(alpha)), with alpha=0.0001).
        """
        return self._eta0

    @eta0.setter
    def eta0(self, value):
        """Sets eta0"""
        self._eta0 = float(value)

    @property
    def power_t(self):
        """The exponent for inverse scaling learning rate."""
        return self._power_t

    @power_t.setter
    def power_t(self, value):
        """Sets power_t"""
        self._power_t = float(value)

    @property
    def kernel(self):
        """Kernel function."""
        return self._kernel

    @property
    def n_tr_samples(self):
        """Returns the number of training samples."""
        return self._tr.shape[0] if self._tr is not None else None

    def _train(self, dataset):
        """Trains the One-Vs-All SGD classifier.

        The following is a private method computing one single
        binary (2-classes) classifier of the OVA schema.

        Representation of each classifier attribute for the multiclass
        case is explained in corresponding property description.

        Parameters
        ----------
        dataset : CDataset
            Binary (2-classes) training set. Must be a :class:`.CDataset`
            instance with patterns data and corresponding labels.

        Returns
        -------
        trained_cls : classifier
            Instance of the used solver trained using input dataset.

        """
        if dataset.num_classes != 2:
            raise ValueError("training can be performed on binary "
                             "(2-classes) datasets only.")

        # Setting up classifier parameters
        sgd = linear_model.SGDClassifier(
            loss=self.loss.class_type,
            penalty=self.regularizer.class_type,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            n_iter=self.n_iter,
            shuffle=self.shuffle,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            class_weight=self.class_weight,
            warm_start=self.warm_start)

        # Pass loss function parameters to classifier
        sgd.set_params(**self.loss.get_params())
        # Pass regularizer function parameters to classifier
        sgd.set_params(**self.regularizer.get_params())

        # Storing training dataset (will be used by discriminant function)
        if self._tr is None:  # Do this once to speed up multiclass
            self._tr = dataset.X

        # Storing the training matrix for kernel mapping
        if self.kernel is None:
            # Training SGD classifier
            sgd.fit(self._tr.get_data(), dataset.Y.tondarray())
        else:
            # Training SGD classifier with kernel mapping
            sgd.fit(CArray(
                self.kernel.k(dataset.X)).get_data(), dataset.Y.tondarray())

        # Temporary storing attributes of trained classifier
        self._w = CArray(sgd.coef_, tosparse=dataset.issparse).ravel()
        if self.fit_intercept is True:
            self._b = CArray(sgd.intercept_)[0]
        else:
            self._b = 0

        return sgd

    def _discriminant_function(self, x, label=1):
        """Compute the distance of the samples in x from the separating hyperplane.

        Discriminant function is always computed wrt positive class.

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
            Flat array of shape (n_patterns,) with discriminant function
            value of each test pattern or scalar if n_patterns == 1.

        """
        # Scores are given by the linear model
        k = x if self.kernel is None else CArray(self.kernel.k(x, self._tr))
        return CClassifierLinear._discriminant_function(self, k)
