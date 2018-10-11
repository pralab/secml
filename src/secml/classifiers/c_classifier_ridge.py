"""
.. module:: CClassifierRidge
   :synopsis: Ridge classifier

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Paolo Russu <paolo.russu@diee.unica.it>

"""
from sklearn.linear_model import RidgeClassifier
from secml.classifiers import CClassifierLinear
from secml.array import CArray
from secml.kernel import CKernel


class CClassifierRidge(CClassifierLinear):
    """Ridge Classifier."""
    class_type = 'ridge'

    def __init__(self, alpha=1.0, kernel=None,
                 max_iter=1e5, class_weight=None, tol=1e-4,
                 fit_intercept=True, normalizer=None):

        # Calling the superclass init
        CClassifierLinear.__init__(self, normalizer=normalizer)

        # Classifier parameters
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.fit_intercept = fit_intercept

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
        return self._tr is None and super(CClassifierRidge, self).is_clear()

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

    def _train(self, dataset):
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
        if self.kernel is None:
            # Training classifier
            ridge.fit(dataset.X.get_data(), dataset.Y.tondarray())
        else:
            # Training SGD classifier with kernel mapping
            ridge.fit(CArray(
                self.kernel.k(dataset.X)).get_data(), dataset.Y.tondarray())

        # Updating global classifier parameters
        self._w = CArray(ridge.coef_, tosparse=dataset.issparse).ravel()
        self._b = CArray(ridge.intercept_)[0] if self.fit_intercept else 0

    def _discriminant_function(self, x, y=1):
        """Computes the distance from the separating hyperplane for each pattern in x.

        The scores are computed in kernel space if kernel is defined.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : {1}
            The label of the class wrt the function should be calculated.
            Discriminant function is always computed wrt positive class (1).

        Returns
        -------
        score : CArray
            Value of the discriminant function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        x = x.atleast_2d()  # Ensuring input is 2-D
        # Compute discriminant function in kernel space if necessary
        k = x if self.kernel is None else CArray(self.kernel.k(x, self._tr))
        # Scores are given by the linear model
        return CClassifierLinear._discriminant_function(self, k, y=y)
