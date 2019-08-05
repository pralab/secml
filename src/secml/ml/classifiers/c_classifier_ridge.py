"""
.. module:: CClassifierRidge
   :synopsis: Ridge classifier

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn.linear_model import RidgeClassifier

from secml.ml.classifiers import CClassifierLinear
from secml.array import CArray
from secml.ml.kernel import CKernel
from secml.ml.classifiers.gradients import CClassifierGradientRidgeMixin
from secml.ml.classifiers.loss import CLossSquare
from secml.ml.classifiers.regularizer import CRegularizerL2
from secml.utils.mixed_utils import check_is_fitted


class CClassifierRidge(CClassifierLinear, CClassifierGradientRidgeMixin):
    """Ridge Classifier.

    Parameters
    ----------
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
        self._kernel = kernel if kernel is None else CKernel.create(kernel)

        self._tr = None  # slot for the training data

    def is_linear(self):
        """Return True if the classifier is linear."""
        if super(CClassifierRidge,
                 self).is_linear() and self.is_kernel_linear():
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
        if self._kernel is not None:
            check_is_fitted(self, '_tr')
        super(CClassifierRidge, self)._check_is_fitted()

    @property
    def kernel(self):
        """Kernel function."""
        return self._kernel

    @kernel.setter
    def kernel(self, kernel):
        """Setting up the Kernel function (None if a linear classifier)."""
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
            # Training SGD classifier with kernel mapping
            ridge.fit(CArray(
                self.kernel.k(dataset.X)).get_data(), dataset.Y.tondarray())

        # Updating global classifier parameters
        self._w = CArray(ridge.coef_, tosparse=dataset.issparse).ravel()
        self._b = CArray(ridge.intercept_)[0] if self.fit_intercept else 0

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
