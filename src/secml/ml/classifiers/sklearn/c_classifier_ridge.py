"""
.. module:: CClassifierRidge
   :synopsis: Ridge classifier

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn.linear_model import RidgeClassifier

from secml.ml.classifiers import CClassifierLinear
from secml.array import CArray
from secml.ml.kernels import CKernel
from secml.ml.classifiers.gradients import CClassifierGradientRidgeMixin
from secml.ml.classifiers.loss import CLossSquare
from secml.ml.classifiers.regularizer import CRegularizerL2
from secml.utils.mixed_utils import check_is_fitted

import warnings


class CClassifierRidge(CClassifierLinear, CClassifierGradientRidgeMixin):
    """Ridge Classifier.

    Parameters
    ----------
    alpha : float, optional
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Default 1.0.
    kernel : None or CKernel subclass, optional

        .. deprecated:: 0.12

        Instance of a CKernel subclass to be used for computing similarity
        between patterns. If None (default), a linear SVM will be created.
        In the future this parameter will be removed from this classifier and
        kernels will have to be passed as preprocess.
    max_iter : int, optional
        Maximum number of iterations for conjugate gradient solver.
        Default 1e5.
    class_weight : {dict, 'balanced', None}, optional
        Set the parameter C of class i to `class_weight[i] * C`.
        If not given (default), all classes are supposed to have
        weight one. The 'balanced' mode uses the values of labels to
        automatically adjust weights inversely proportional to
        class frequencies as `n_samples / (n_classes * np.bincount(y))`.
    tol : float, optional
        Precision of the solution. Default 1e-4.
    fit_intercept : bool, optional
        If True (default), the intercept is calculated, else no intercept will
        be used in calculations (e.g. data is expected to be already centered).
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'ridge'

    """
    __class_type = 'ridge'

    _loss = CLossSquare()
    _reg = CRegularizerL2()

    def __init__(self, alpha=1.0, kernel=None,
                 max_iter=1e5, class_weight=None, tol=1e-4,
                 fit_intercept=True, preprocess=None):

        # Calling the superclass init
        CClassifierLinear.__init__(self, preprocess=preprocess)

        # Classifier parameters
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.fit_intercept = fit_intercept

        # Similarity function (bound) to use for computing features
        # Keep private (not a param of RIDGE)
        if kernel is not None:
            warnings.warn("`kernel` parameter in `CClassifierRidge` is "
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
        if self._kernel is not None:
            check_is_fitted(self, '_tr')
        super(CClassifierRidge, self)._check_is_fitted()

    @property
    def kernel(self):
        """Kernel function."""
        return self._kernel

    @kernel.setter
    def kernel(self, kernel):
        """Setting up the Kernel function (None if a linear classifier).
        This property is deprecated, as in the future kernel will have to be
        passed as preprocess."""
        warnings.warn("`kernel` parameter in `CClassifierRidge` is "
                      "deprecated from 0.12, in the future kernels will "
                      "have to be passed as preprocess.", DeprecationWarning)
        self._kernel = kernel

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
    def tr(self):
        """Training set."""
        return self._tr

    @property
    def n_tr_samples(self):
        """Returns the number of training samples."""
        return self._tr.shape[0] if self._tr is not None else None

    def _fit(self, dataset):
        """Trains the One-Vs-All Ridge classifier.

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
        ridge = RidgeClassifier(alpha=self.alpha,
                                fit_intercept=self.fit_intercept,
                                tol=self.tol,
                                max_iter=self.max_iter,
                                class_weight=self.class_weight,
                                solver='auto')

        # Storing training dataset (only if required by kernel)
        self._tr = dataset.X if self._kernel is not None else None

        # Storing the training matrix for kernel mapping
        if self.is_kernel_linear():
            # Training classifier
            ridge.fit(dataset.X.get_data(), dataset.Y.tondarray())
        else:
            # Training Ridge classifier with kernel mapping
            ridge.fit(CArray(
                self.kernel.k(dataset.X)).get_data(), dataset.Y.tondarray())

        # Updating global classifier parameters
        self._w = CArray(ridge.coef_, tosparse=dataset.issparse).ravel()
        self._b = CArray(ridge.intercept_)[0] if self.fit_intercept else 0

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
        x : CArray or None, optional
            The gradient is computed in the neighborhood of x.
        y : int, optional
            Binary index of the class wrt the gradient must be computed.
            Default is 1, corresponding to the positive class.

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
