"""
.. module:: CClassifierSVM
   :synopsis: Support Vector Machine (SVM) classifier

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from sklearn.svm import SVC

from secml.array import CArray
from secml.ml.classifiers import CClassifier
from secml.ml.classifiers.clf_utils import convert_binary_labels
from secml.ml.kernels import CKernel
from secml.ml.classifiers.gradients import CClassifierGradientSVMMixin
from secml.ml.classifiers.loss import CLossHinge
from secml.utils.mixed_utils import check_is_fitted


class CClassifierSVMM(CClassifier):
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
    __class_type = 'svm-lin'

    _loss = CLossHinge()

    def __init__(self, C=1.0, kernel='linear', preprocess=None):
        # Calling the superclass init
        CClassifier.__init__(self, preprocess=preprocess)

        # Classifier hyperparameters
        self.C = C
        self.kernel = kernel

        # After-training attributes
        self._w = None
        self._b = None
        self._alpha = None

    def _is_kernel_linear(self):
        if self._kernel.__class__ == 'linear':
            return True
        else:
            return False

    @property
    def kernel(self):
        """Kernel function (None if a linear classifier)."""
        return self._kernel

    @kernel.setter
    def kernel(self, kernel):
        """Setting up the Kernel function (None if a linear classifier)."""
        self._kernel = CKernel.create(kernel)

        if not self._is_kernel_linear():
            # set kernel as preprocessor for the current classifier
            preprocess = self.preprocess
            self.preprocess = self.kernel
            self.kernel.preprocess = preprocess

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b

    @property
    def alpha(self):
        """Signed coefficients of the SVs in the decision function."""
        return self._alpha

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

    def _fit(self, x, y):
        """Trains the One-Vs-All SVM classifier.

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features).
        y : CArray
            Array of shape (n_samples,) containing the class
            labels (2-classes only).

        Returns
        -------
        CClassifierSVM
            Trained classifier.

        """
        self.logger.info(
            "Training SVM with parameters: {:}".format(self.get_params()))

        # reset training
        self._w = None
        self._b = None
        self._alpha = None

        # shape of w or alpha
        n_rows = self.n_classes if self.n_classes > 2 else 1
        n_cols = x.shape[1]

        # initialize params
        if self._is_kernel_linear():
            svc_kernel = 'linear'
            self._w = CArray.zeros(shape=(n_rows, n_cols))
        else:
            svc_kernel = 'precomputed'
            self._alpha = CArray.zeros(shape=(n_rows, n_cols), sparse=True)
        self._b = CArray.zeros(shape=(self.n_classes,))

        if self.n_classes > 2:
            # fit OVA
            self._fit_OVA(x, y, svc_kernel)
        else:
            # fit binary
            classifier = SVC(C=self.C, kernel=svc_kernel)
            classifier.fit(x.get_data(), y.get_data())
            if self._is_kernel_linear():
                self._w = CArray(classifier.coef_.ravel())
            else:
                sv_idx = CArray(classifier.support_).ravel()
                self._alpha[sv_idx] = CArray(classifier.dual_coef_)
            self._b = CArray(classifier.intercept_[0])[0]

        # remove unused support vectors from kernel
        if self._alpha is not None:  # trained in the dual
            sv = abs(self._alpha).sum(axis=0) > 0
            self.kernel.rv = self.kernel.rv[sv, :]
            self._alpha = self._alpha[:, sv]

        return self

    def _fit_OVA(self, x, y, svc_kernel):
        # ova (but we can also implement ovo - let's do separate functions)
        for k, c in enumerate(self.classes):
            classifier = SVC(C=self.C, kernel=svc_kernel)
            classifier.fit(x.get_data(), CArray(y == c).get_data())
            if self._is_kernel_linear():
                self._w[k, :] = CArray(classifier.coef_.ravel())
            else:
                sv_idx = CArray(classifier.support_).ravel()
                self._alpha[k, sv_idx] = CArray(classifier.dual_coef_)
            self._b[k] = CArray(classifier.intercept_[0])[0]
        return

    def _forward(self, x):
        """Compute decision function for SVMs, proportional to the distance of
        x to the separating hyperplane.

        For non linear SVM, the kernel between input patterns and
         Support Vectors is computed and then the inner product of
         the resulting array with the alphas is calculated.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features) or (n_patterns, n_sv) if kernel is used.

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_samples,) if `y` is not None,
            otherwise a (n_samples, n_classes) array.

        """
        v = self.w if self._is_kernel_linear() else self.alpha
        score = CArray(x.dot(v.T)) + self.b
        if self.n_classes > 2:  # return current score matrix
            scores = score
        else:  # concatenate scores
            scores = CArray.ones(shape=(x.shape[0], self.n_classes))
            scores[:, 0] = -score.ravel().T
            scores[:, 1] = score.ravel().T
        return scores

    def _backward(self, w):
        v = self.w if self._is_kernel_linear() else self.alpha
        if self.n_classes > 2:
            return w.dot(v)
        else:
            return w[0] * -v + w[1] * v
