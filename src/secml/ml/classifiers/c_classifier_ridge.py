"""
.. module:: CClassifierRidge
   :synopsis: Ridge classifier

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Paolo Russu <paolo.russu@diee.unica.it>

"""
from sklearn.linear_model import RidgeClassifier

from secml.ml.classifiers import CClassifierLinear
from secml.ml.classifiers.clf_utils import convert_binary_labels
from secml.array import CArray
from secml.ml.kernel import CKernel
from secml.ml.classifiers.gradients import CClassifierGradientRidge


class CClassifierRidge(CClassifierLinear):
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

        self._gradients = CClassifierGradientRidge()

    def __clear(self):
        """Reset the object."""
        self._tr = None

    def __is_clear(self):
        """Returns True if object is clear."""
        if self._tr is not None:
            return False

        # CClassifierLinear attributes
        if self._w is not None or self._b is not None:
            return False

        return True

    def is_linear(self):
        """Return True if the classifier is linear."""
        if super(CClassifierRidge, self).is_linear() and self.is_kernel_linear():
            return True
        return False

    def is_kernel_linear(self):
        """Return True if the kernel is None or linear."""
        if self.kernel is None or self.kernel.class_type == 'linear':
            return True
        return False

    @property
    def gradients(self):
        return self._gradients

    @property
    def kernel(self):
        """Kernel function."""
        return self._kernel

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
        if self._kernel is not None:
            self._tr = dataset.X

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

    def _decision_function(self, x, y=1):
        """Computes the distance from the separating hyperplane for each pattern in x.

        The scores are computed in kernel space if kernel is defined.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : {1}
            The label of the class wrt the function should be calculated.
            decision function is always computed wrt positive class (1).

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        x = x.atleast_2d()  # Ensuring input is 2-D
        # Compute decision function in kernel space if necessary
        k = x if self.is_kernel_linear() else CArray(self.kernel.k(x, self._tr))
        # Scores are given by the linear model
        return CClassifierLinear._decision_function(self, k, y=y)

    def _gradient_f(self, x=None, y=1):
        """Computes the gradient of the linear classifier's decision function
         wrt decision function input.

        For linear classifiers, the gradient wrt input is equal
        to the weights vector w. The point x can be in fact ignored.

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
        x = x.atleast_2d()

        if self.is_kernel_linear():  # Simply return w for a linear Ridge
            return CClassifierLinear._gradient_f(self, y=y)

        # Point is required in the case of non-linear Ridge
        if x is None:
            raise ValueError("point 'x' is required to compute the gradient")

        gradient = self.kernel.gradient(self._tr, x).atleast_2d()

        # Few shape check to ensure broadcasting works correctly
        if gradient.shape != (self._tr.shape[0], self.n_features):
            raise ValueError("Gradient shape must be ({:}, {:})".format(
                x.shape[0], self.n_features))

        w_2d = self.w.atleast_2d()
        if gradient.issparse is True:  # To ensure the sparse dot is used
            w_2d = w_2d.tosparse()
        if w_2d.shape != (1, self._tr.shape[0]):
            raise ValueError("Weight vector shape must be ({:}, {:}) "
                             "or ravel equivalent".format(1, self._tr.shape[0]))

        gradient = w_2d.dot(gradient)

        # Gradient sign depends on input label (0/1)
        return convert_binary_labels(y) * gradient.ravel()

