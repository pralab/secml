"""
.. module:: CClassifierSGD
   :synopsis: Stochastic Gradient Descent (SGD) classifier

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn import linear_model

from secml.array import CArray
from secml.core.constants import inf
from secml.ml.classifiers import CClassifierLinear
from secml.ml.classifiers.loss import CLoss
from secml.ml.classifiers.regularizer import CRegularizer
from secml.ml.kernel import CKernel
from secml.utils.mixed_utils import check_is_fitted
from secml.ml.classifiers.gradients import CClassifierGradientSGDMixin


class CClassifierSGD(CClassifierLinear, CClassifierGradientSGDMixin):
    """Stochastic Gradient Descent Classifier.

    Parameters
    ----------
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'sgd'

    """
    __class_type = 'sgd'

    def __init__(self, loss, regularizer, kernel=None, alpha=0.01,
                 fit_intercept=True, max_iter=1000, tol=-inf,
                 shuffle=True, learning_rate='optimal',
                 eta0=10.0, power_t=0.5, class_weight=None,
                 warm_start=False, average=False, random_state=None,
                 preprocess=None):

        # Calling the superclass init
        CClassifierLinear.__init__(self, preprocess=preprocess)

        # Keep private (not an sklearn sgd parameter)
        self._loss = CLoss.create(loss)
        # Keep private (not an sklearn sgd parameter)
        self._regularizer = CRegularizer.create(regularizer)

        # Classifier parameters
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol  # TODO: from sklearn 0.21 default for tol will change
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = average
        self.random_state = random_state

        # Similarity function (bound) to use for computing features
        # Keep private (not a param of SGD)
        self._kernel = kernel if kernel is None else CKernel.create(kernel)

        self._tr = None  # slot for the training data

    def is_linear(self):
        """Return True if the classifier is linear."""
        if super(CClassifierSGD, self).is_linear() and self.is_kernel_linear():
            return True
        return False

    def is_kernel_linear(self):
        """Return True if the kernel is None or linear."""
        if self.kernel is None or self.kernel.class_type == 'linear':
            return True
        return False

    def _check_is_fitted(self):
        """Check if the classifier is trained (fitted).

        Raises
        ------
        NotFittedError
            If the classifier is not fitted.

        """
        if not self.is_kernel_linear():
            check_is_fitted(self, '_tr')
        super(CClassifierSGD, self)._check_is_fitted()

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
    def C(self):
        """Constant that multiplies the regularization term.

        Equal to 1 / alpha.

        """
        return 1.0 / self.alpha

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
    def average(self):
        """When set to True, computes the averaged SGD weights.
        If set to an int greater than 1, averaging will begin once the total
        number of samples seen reaches average.
        So average=10 will begin averaging after seeing 10 samples."""
        return self._average

    @average.setter
    def average(self, value):
        """Sets the average parameter."""
        self._average = value

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

    @kernel.setter
    def kernel(self, kernel):
        """Setting up the Kernel function (None if a linear classifier)."""
        self._kernel = kernel

    @property
    def n_tr_samples(self):
        """Returns the number of training samples."""
        return self._tr.shape[0] if self._tr is not None else None

    def _fit(self, dataset):
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
        # TODO: remove this check from here.
        #  It should be handled in CClassifierLinear
        if dataset.num_classes != 2:
            raise ValueError("training can be performed on binary "
                             "(2-classes) datasets only.")

        # TODO: remove this object init from here. Put into constructor
        #  See RandomForest / DT and inherit from CClassifierSKlearn
        # Setting up classifier parameters
        sgd = linear_model.SGDClassifier(
            loss=self.loss.class_type,
            penalty=self.regularizer.class_type,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=self.shuffle,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            class_weight=self.class_weight,
            average=self.average,
            warm_start=self.warm_start,
            random_state=self.random_state)

        # Pass loss function parameters to classifier
        sgd.set_params(**self.loss.get_params())
        # Pass regularizer function parameters to classifier
        sgd.set_params(**self.regularizer.get_params())

        # TODO: remove unconventional kernel usage. This is a linear classifier
        # Storing training dataset (will be used by decision function)
        self._tr = dataset.X if not self.is_kernel_linear() else None

        # Storing the training matrix for kernel mapping
        if self.is_kernel_linear():
            # Training SGD classifier
            sgd.fit(dataset.X.get_data(), dataset.Y.tondarray())
        else:
            # Training SGD classifier with kernel mapping
            sgd.fit(CArray(
                self.kernel.k(dataset.X)).get_data(), dataset.Y.tondarray())

        # Temporary storing attributes of trained classifier
        self._w = CArray(sgd.coef_, tosparse=dataset.issparse).ravel()
        if self.fit_intercept is True:
            self._b = CArray(sgd.intercept_)[0]
        else:
            self._b = None

        return sgd

    # TODO: this function can be removed when removing kernel support
    def _decision_function(self, x, y=None):
        """Computes the distance from the separating hyperplane for each pattern in x.

        The scores are computed in kernel space if kernel is defined.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : {0, 1, None}
            The label of the class wrt the function should be calculated.
            If None, return the output for all classes.

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_samples,) if `y` is not None,
            otherwise a (n_samples, n_classes) array.

        """
        # Compute decision function in kernel space if necessary
        k = x if self.is_kernel_linear() else \
            CArray(self.kernel.k(x, self._tr))
        # Scores are given by the linear model
        return CClassifierLinear._decision_function(self, k, y=y)


