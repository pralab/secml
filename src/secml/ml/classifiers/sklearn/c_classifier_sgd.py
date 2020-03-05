"""
.. module:: CClassifierSGD
   :synopsis: Stochastic Gradient Descent (SGD) classifier

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn import linear_model

from secml.array import CArray
from secml.ml.classifiers import CClassifierLinear
from secml.ml.classifiers.loss import CLoss
from secml.ml.classifiers.regularizer import CRegularizer
from secml.ml.kernels import CKernel
from secml.utils.mixed_utils import check_is_fitted
from secml.ml.classifiers.gradients import CClassifierGradientSGDMixin

import warnings


class CClassifierSGD(CClassifierLinear, CClassifierGradientSGDMixin):
    """Stochastic Gradient Descent Classifier.

    Parameters
    ----------
    loss : CLoss
        Loss function to be used during classifier training.
    regularizer : CRegularizer
        Regularizer function to be used during classifier training.
    kernel : None or CKernel subclass, optional

        .. deprecated:: 0.12

        Instance of a CKernel subclass to be used for computing similarity
        between patterns. If None (default), a linear SVM will be created.
        In the future this parameter will be removed from this classifier and
        kernels will have to be passed as preprocess.
    alpha : float, optional
        Constant that multiplies the regularization term. Default 0.01.
        Also used to compute learning_rate when set to 'optimal'.
    fit_intercept : bool, optional
        If True (default), the intercept is calculated, else no intercept will
        be used in calculations (e.g. data is expected to be already centered).
    max_iter : int, optional
        The maximum number of passes over the training data (aka epochs).
        Default 1000.
    tol : float or None, optional
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > best_loss - tol) for 5 consecutive epochs. Default None.
    shuffle : bool, optional
        If True (default) the training data is shuffled after each epoch.
    learning_rate : str, optional
        The learning rate schedule. If 'constant', eta = eta0;
        if 'optimal' (default), eta = 1.0 / (alpha * (t + t0)), where t0 is
        chosen by a heuristic proposed by Leon Bottou; if 'invscaling',
        eta = eta0 / pow(t, power_t); if 'adaptive', eta = eta0, as long as
        the training keeps decreasing.
    eta0 : float, optional
        The initial learning rate for the 'constant', 'invscaling' or
        'adaptive' schedules. Default 10.0.
    power_t : float, optional
        The exponent for inverse scaling learning rate. Default 0.5.
    class_weight : {dict, 'balanced', None}, optional
        Set the parameter C of class i to `class_weight[i] * C`.
        If not given (default), all classes are supposed to have
        weight one. The 'balanced' mode uses the values of labels to
        automatically adjust weights inversely proportional to
        class frequencies as `n_samples / (n_classes * np.bincount(y))`.
    warm_start : bool, optional
        If True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Default False.
    average : bool or int, optional
        If True, computes the averaged SGD weights and stores the result in
        the `coef_` attribute. If set to an int greater than 1, averaging
        will begin once the total number of samples seen reaches average.
        Default False.
    random_state : int, RandomState or None, optional
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Default None.
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
                 fit_intercept=True, max_iter=1000, tol=None,
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
        self.tol = tol
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
        if kernel is not None:
            warnings.warn("`kernel` parameter in `CClassifierSGD` is "
                          "deprecated from 0.12, in the future kernels will "
                          "have to be passed as preprocess.",
                          DeprecationWarning)
        self._kernel = kernel if kernel is None else CKernel.create(kernel)

        self._tr = None  # slot for the training data

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
        Default is 10.0 (corresponding to sqrt(1.0/sqrt(alpha)),
        with alpha=0.0001).
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
        """Setting up the Kernel function (None if a linear classifier).
        This property is deprecated, as in the future kernel will have to be
        passed as preprocess."""
        warnings.warn("`kernel` parameter in `CClassifierSGD` is "
                      "deprecated from 0.12, in the future kernels will "
                      "have to be passed as preprocess.", DeprecationWarning)
        self._kernel = kernel

    @property
    def tr(self):
        """Training set."""
        return self._tr

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
    def _forward(self, x):
        """Computes the distance from the separating hyperplane
        for each pattern in x.

        The scores are computed in kernel space if kernel is defined.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).

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
        return CClassifierLinear._forward(self, k)

    def _backward(self, w=None):
        """Computes the gradient of the linear classifier's decision function
         wrt decision function input.

        For linear classifiers, the gradient wrt the input x is equal
        to the weight vector w, regardless of x.

        Parameters
        ----------
        w : CArray or None
            if CArray, it is pre-multiplied to the gradient
            of the module, as in standard reverse-mode autodiff.

        Returns
        -------
        gradient : CArray
            The gradient of the linear classifier's decision function
            wrt decision function input. Vector-like array.

        """
        if self.is_kernel_linear():  # Simply return w for a linear Ridge
            gradient = self.w.ravel()
        else:
            self.kernel.reference_samples = self._tr
            gradient = self.kernel.gradient(self._cached_x).atleast_2d()

            # Few shape check to ensure broadcasting works correctly
            if gradient.shape != (self._tr.shape[0], self.n_features):
                raise ValueError("Gradient shape must be ({:}, {:})".format(
                    self._cached_x.shape[0], self.n_features))

            w_2d = self.w.atleast_2d()
            if gradient.issparse is True:  # To ensure the sparse dot is used
                w_2d = w_2d.tosparse()
            if w_2d.shape != (1, self._tr.shape[0]):
                raise ValueError(
                    "Weight vector shape must be ({:}, {:}) "
                    "or ravel equivalent".format(1, self._tr.shape[0]))

            gradient = w_2d.dot(gradient)

        # Gradient sign depends on input label (0/1)
        if w is not None:
            return w[0] * -gradient + w[1] * gradient
        else:
            raise ValueError("w cannot be set as None.")
