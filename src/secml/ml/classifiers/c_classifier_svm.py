"""
.. module:: CClassifierSVM
   :synopsis: Support Vector Machine (SVM) classifier

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from sklearn.svm import SVC

from secml.array import CArray
from secml.ml.classifiers import CClassifierLinear
from secml.ml.classifiers.clf_utils import convert_binary_labels
from secml.ml.kernel import CKernel
from secml.ml.classifiers.gradients import CClassifierGradientSVMMixin
from secml.ml.classifiers.loss import CLossHinge
from secml.utils.mixed_utils import check_is_fitted


class CClassifierSVM(CClassifierLinear, CClassifierGradientSVMMixin):
    """Support Vector Machine (SVM) classifier.

    Parameters
    ----------
    kernel : None or CKernel subclass, optional
        Instance of a CKernel subclass to be used for computing
        similarity between patterns. If None (default), a linear
        SVM will be created.
    C : float, optional
        Penalty parameter C of the error term. Default 1.0.
    class_weight : {dict, 'balanced', None}, optional
        Set the parameter C of class i to `class_weight[i] * C`.
        If not given (default), all classes are supposed to have
        weight one. The 'balanced' mode uses the values of labels to
        automatically adjust weights inversely proportional to
        class frequencies as `n_samples / (n_classes * np.bincount(y))`.
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.
    grad_sampling : float
        Percentage in (0.0, 1.0] of the alpha weights to be considered
        when computing the classifier gradient.

    Attributes
    ----------
    class_type : 'svm'

    Notes
    -----
    Current implementation relies on :class:`sklearn.svm.SVC` for
    the training step.

    See Also
    --------
    CKernel : Pairwise kernels and metrics.
    CClassifierLinear : Common interface for linear classifiers.

    """
    __class_type = 'svm'

    _loss = CLossHinge()

    def __init__(self, kernel=None, C=1.0, class_weight=None,
                 preprocess=None, grad_sampling=1.0, store_dual_vars=None):

        # Calling the superclass init
        CClassifierLinear.__init__(self, preprocess=preprocess)

        # Classifier parameters
        self.C = C
        self.class_weight = class_weight
        # Number of samples for approx. gradient
        self.grad_sampling = grad_sampling

        # Flag that control storing of dual variables (depends on kernel)
        self._store_dual_vars = store_dual_vars

        # Setting up the kernel function
        self.kernel = CKernel.create('linear') if kernel is None \
            else CKernel.create(kernel)

        # After-training attributes
        self._n_sv = None
        self._sv_idx = None
        self._alpha = None
        self._sv = None

        # slot for the computed kernel function (to speed up multiclass)
        # DO NOT CLEAR
        self._k = None

    def is_linear(self):
        """Return True if the classifier is linear."""
        if super(CClassifierSVM, self).is_linear() and self.is_kernel_linear():
            return True
        return False

    def is_kernel_linear(self):
        """Return True if the kernel is None or linear."""
        if self.kernel.class_type == 'linear':
            return True
        return False

    def _check_is_fitted(self):
        """Check if the classifier is trained (fitted).

        Raises
        ------
        NotFittedError
            If the classifier is not fitted.

        """
        if not self.is_kernel_linear() or self.store_dual_vars is True:
            check_is_fitted(self, 'sv')  # Checking the SVs is enough
        # SVM is a special case, is not set '_w' if kernel is not linear
        # so we cannot call the superclass `_check_is_fitted`
        if self.is_kernel_linear():
            check_is_fitted(self, 'w')
        # Then check the attributes of CClassifier
        check_is_fitted(self, ['classes', 'n_features'])

    @property
    def C(self):
        """Penalty parameter C of the error term."""
        return self._C

    @C.setter
    def C(self, value):
        """Set the penalty parameter C of the error term.

        Parameters
        ----------
        value : float
            Penalty parameter C of the error term.

        """
        self._C = float(value)

    @property
    def class_weight(self):
        """Weight of each training class."""
        return self._class_weight

    @class_weight.setter
    def class_weight(self, value):
        """Sets the weight of each training class.

        Parameters
        ----------
        value : {dict, 'balanced', None}
            Set the parameter C of class i to `class_weight[i] * C`.
            If None, all classes are supposed to have weight one.
            The 'auto' mode uses the values of labels to automatically
             adjust weights inversely proportional to class frequencies
             as `n_samples / (n_classes * np.bincount(y))`.

        """
        if isinstance(value, dict) and len(value) != 2:
            raise ValueError("weight of positive (+1) and negative (0) "
                             "classes only must be specified.")
        self._class_weight = value

    @property
    def kernel(self):
        """Kernel function (None if a linear classifier)."""
        return self._kernel

    @kernel.setter
    def kernel(self, kernel_obj):
        """Setting up the Kernel function (None if a linear classifier)."""
        self._kernel = kernel_obj
        # Check store dual variables flag after kernel change
        self.store_dual_vars = self.store_dual_vars

    @property
    def grad_sampling(self):
        """Percentage of samples for approximate gradient."""
        return self._grad_sampling

    @grad_sampling.setter
    def grad_sampling(self, value):
        """Percentage of samples for approximate gradient."""
        self._grad_sampling = value

    @property
    def store_dual_vars(self):
        """Controls the store of dual space variables (SVs and alphas).

        By default is None and dual variables are stored only if
        kernel is not None. If set to True, dual variables are
        stored even if kernel is None  (linear SVM). If kernel
        is not None, cannot be set to False.

        """
        return self._store_dual_vars

    @store_dual_vars.setter
    def store_dual_vars(self, value):
        """Controls the store of dual space variables (SVs and alphas).

        Parameters
        ----------
        value : bool or None
            By default is None and dual variables are stored only if
            kernel is not None. If set to True, dual variables are
            stored even if kernel is None (linear SVM). If kernel
            is not None, cannot be set to False.

        """
        if value is not None:
            if not self.is_kernel_linear() and value is False:
                raise ValueError(
                    "not linear SVM, dual variables are always stored. "
                    "Set store_dual_vars to None or True.")
        self._store_dual_vars = value

    @property
    def alpha(self):
        """Signed coefficients of the SVs in the decision function."""
        return self._alpha

    @property
    def n_sv(self):
        """Return the number of support vectors.

        In the 1st and in the 2nd column is stored the number
        of SVs for the negative and positive class respectively.

        """
        return self._n_sv

    @property
    def sv_idx(self):
        """Indices of Support Vectors within the training dataset."""
        return self._sv_idx

    @property
    def sv(self):
        """Support Vectors."""
        return self._sv

    def sv_margin_idx(self, tol=1e-6):
        """Indices of Margin Support Vectors.

        Parameters
        ----------
        tol : float
            Alpha value threshold for considering a
            Support Vector on the margin.

        Returns
        -------
        indices : CArray
            Flat array with the indices of the Margin Support Vectors.

        """
        s = self.alpha.find(
            (abs(self.alpha) >= tol) *
            (abs(self.alpha) <= self.C - tol))
        return CArray(s)

    def sv_margin(self, tol=1e-6):
        """Margin Support Vectors.

        Parameters
        ----------
        tol : float
            Alpha value threshold for considering a
            Support Vector on the margin.

        Returns
        -------
        CArray or None
            Margin support vector, 2D CArray.
            If no margin support vector are found, return None.
        indices : CArray or None
            Flat array with the indices of the margin support vectors.
            If no margin support vector are found, return None.

        """
        s = self.sv_margin_idx(tol=tol)

        if s.size == 0:
            return None, None

        xs = self.sv[s, :].atleast_2d()
        return xs, s

    def sv_margin_y(self, tol=1e-6):
        """Margin Support Vectors class (-1/+1).

        Parameters
        ----------
        tol : float
            Alpha value threshold for considering a
            Support Vector on the margin.

        Returns
        -------
        CArray
            Flat CArray with the class (-1/+1) of the Margin Support Vectors.

        """
        ys = self.alpha.sign()
        return ys[self.sv_margin_idx(tol=tol)]

    def fit(self, dataset, n_jobs=1):
        """Fit the SVM classifier.

        We use :class:`sklearn.svm.SVC` for weights and Support Vectors
        computation. The routine will set alpha, sv, sv_idx and b
        parameters. For linear SVM (i.e. if kernel is None)
        we also store the 'w' flat vector with each feature's weight.

        If a preprocess has been specified, input is normalized
        before computing the decision function.

        Parameters
        ----------
        dataset : CDataset
            Binary (2-classes) Training set. Must be a :class:`.CDataset`
            instance with patterns data and corresponding labels.
        n_jobs : int, optional
            Number of parallel workers to use for training the classifier.
            Default 1. Cannot be higher than processor's number of cores.

        Returns
        -------
        trained_cls : CClassifierSVM
            Instance of the SVM classifier trained using input dataset.

        """
        super(CClassifierSVM, self).fit(dataset, n_jobs=n_jobs)
        # Cleaning up kernel matrix to free memory
        self._k = None

        return self

    def _fit(self, dataset):
        """Trains the One-Vs-All SVM classifier.

        Parameters
        ----------
        dataset : CDataset
            Binary (2-classes) training set. Must be a :class:`.CDataset`
            instance with patterns data and corresponding labels.

        Returns
        -------
        trained_cls : CCLassifierSVM
            Instance of the SVM classifier trained using input dataset.

        """
        self.logger.info(
            "Training SVM with parameters: {:}".format(self.get_params()))
        # Setting up classifier parameters
        classifier = SVC(C=self.C, class_weight=self.class_weight,
                         kernel='linear' if self.is_kernel_linear()
                         else 'precomputed')

        # Computing the kernel matrix
        if not self.is_kernel_linear():
            self._k = CArray(self.kernel.k(dataset.X))
        else:
            self._k = dataset.X

        # Training classifier using precomputed kernel
        classifier.fit(self._k.get_data(), dataset.Y.tondarray())

        # Intercept
        self._b = CArray(classifier.intercept_[0])[0]
        self.logger.debug("Classifier SVM bias: {:}".format(self._b))

        # Updating SVM parameters
        self._w = None  # Resetting `_w` to leave it None next cond is False
        if self.is_kernel_linear():  # Linear SVM
            self._w = CArray(
                CArray(classifier.coef_, tosparse=dataset.issparse).ravel())
            self.logger.debug(
                "Classifier SVM linear weights: \n{:}".format(self._w))

        if not self.is_kernel_linear() or self.store_dual_vars is True:
            # Dual Space SVM or forced dual variables store
            self._n_sv = CArray(classifier.n_support_)
            self._sv_idx = CArray(classifier.support_).ravel()
            # Compatibility fix for differences between sklearn versions
            self._alpha = convert_binary_labels(dataset.Y[self.sv_idx]) * \
                          abs(CArray(classifier.dual_coef_).todense().ravel())
            self._sv = CArray(dataset.X[self.sv_idx, :])
            self.logger.debug("Classifier SVM dual weights (alphas): "
                              "\n{:}".format(self._alpha))
        else:  # Resetting the dual parameters
            self._n_sv = None
            self._sv_idx = None
            self._alpha = None
            self._sv = None

        return classifier

    def _decision_function(self, x, y=None):
        """Computes the distance from the separating hyperplane for each pattern in x.

        For non linear SVM, the kernel between input patterns and
         Support Vectors is computed and then the inner product of
         the resulting array with the alphas is calculated.

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
        if y not in (0, 1, None):
            raise ValueError("decision function cannot be computed "
                             "against class {:}.".format(y))

        if self.is_kernel_linear():  # Scores are given by the linear model
            return CClassifierLinear._decision_function(self, x, y=y)

        k = CArray(self.kernel.k(x, self.sv)).dot(self.alpha.T)
        score = CArray(k).todense().ravel() + self.b

        scores = CArray.ones(shape=(x.shape[0], self.n_classes))
        scores[:, 0] = -score.ravel().T
        scores[:, 1] = score.ravel().T

        return scores[:, y].ravel() if y is not None else scores

