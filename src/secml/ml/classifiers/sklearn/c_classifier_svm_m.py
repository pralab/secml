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


class CClassifierSVM(CClassifier):
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
    preprocess : CModule or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'svm-m'

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

    def __init__(self, kernel='linear', C=1.0, class_weight=None,
                 preprocess=None, store_dual_vars=None):

        # calling the superclass init
        CClassifier.__init__(self, preprocess=preprocess)

        # Classifier hyperparameters
        self.C = C
        self.class_weight = class_weight

        # After-training attributes
        self._w = None
        self._b = None
        self._alpha = None
        self._sv_idx = None  # idx of SVs in TR data (only for binary SVM)

        self._store_dual_vars = bool(store_dual_vars)

        self._kernel = None
        self.kernel = kernel

    def _is_kernelized(self):
        """Return True if SVM has to be trained in the dual space."""
        if self._kernel.class_type == 'linear' and not self.store_dual_vars:
            return False
        else:
            return True

    @property
    def sv_idx(self):
        """Indices of Support Vectors within the training dataset."""
        return self._sv_idx

    @property
    def kernel(self):
        """Kernel function."""
        return self._kernel

    @kernel.setter
    def kernel(self, kernel):
        """Set the kernel function."""
        # extract preprocess attached to kernel, if any
        if self.preprocess is self.kernel:  # if point to the same ref
            p = self.kernel.preprocess if self.kernel is not None else None
        else:
            p = self.preprocess

        kernel = 'linear' if kernel is None else kernel
        self._kernel = CKernel.create(kernel)

        if self._is_kernelized():
            # set kernel as preprocessor for the current classifier
            # and train classifier in the dual (using the precomputed kernel)
            self.preprocess = self.kernel
            # connect new kernel to previous pre-processing chain
            self.kernel.preprocess = p
        else:
            self.preprocess = p  # bypass (linear) kernel

    @property
    def store_dual_vars(self):
        """Train the SVM classifier in the dual.

        By default is None or False.
        If True, the SVM is trained in the dual even if the kernel is linear.
        For nonlinear kernels, the SVM is always trained in the dual
        (this value is ignored).

        """
        return self._store_dual_vars

    @store_dual_vars.setter
    def store_dual_vars(self, value):
        """Train the SVM classifier in the dual.

        Parameters
        ----------
        value : bool
            By default is None or False.
            If True, the SVM is trained in the dual even if the kernel
            is linear. For nonlinear kernels, the SVM is always trained
            in the dual(this value is ignored).

        """
        self._store_dual_vars = bool(value)
        if self.kernel.class_type == 'linear':
            self.kernel = self.kernel  # need to reset kernel and preprocessing

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
        # TODO we can have one weight per class but only for OVO
        if isinstance(value, dict) and len(value) != 2:
            raise ValueError("weight of positive (+1) and negative (0) "
                             "classes only must be specified.")
        self._class_weight = value

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
        self._sv_idx = None

        # shape of w or alpha
        n_rows = self.n_classes if self.n_classes > 2 else 1
        n_cols = x.shape[1]

        # initialize params
        if not self._is_kernelized():
            # no kernel preprocessing, training in the primal
            svc_kernel = 'linear'
            self._w = CArray.zeros(shape=(n_rows, n_cols))
        else:
            # inputs are kernel values, training in the dual
            svc_kernel = 'precomputed'
            self._alpha = CArray.zeros(shape=(n_rows, n_cols), sparse=True)
        self._b = CArray.zeros(shape=(self.n_classes,))

        if self.n_classes > 2:
            # fit OVA
            self._fit_one_vs_all(x, y, svc_kernel)
        else:
            # fit binary
            self._fit_binary(x, y, svc_kernel)

        # remove unused support vectors from kernel
        if self._is_kernelized():  # trained in the dual
            sv = abs(self._alpha).sum(axis=0) > 0
            self.kernel.rv = self.kernel.rv[sv, :]
            self._alpha = self._alpha[:, sv]

        assert (self.kernel.rv.shape[0] == self.alpha.shape[1])

        return self

    def _fit_one_vs_all(self, x, y, svc_kernel):
        # ova (but we can also implement ovo - let's do separate functions)
        for k, c in enumerate(self.classes):
            # TODO: class weights - balanced by default?
            classifier = SVC(C=self.C, kernel=svc_kernel,
                             class_weight=self.class_weight)
            classifier.fit(x.get_data(), CArray(y == c).get_data())
            if not self._is_kernelized():
                self._w[k, :] = CArray(classifier.coef_.ravel())
            else:
                sv_idx = CArray(classifier.support_).ravel()
                self._alpha[k, sv_idx] = CArray(classifier.dual_coef_)
            self._b[k] = CArray(classifier.intercept_[0])[0]
        return

    def _fit_binary(self, x, y, svc_kernel):
        classifier = SVC(C=self.C, kernel=svc_kernel,
                         class_weight=self.class_weight)
        classifier.fit(x.get_data(), y.get_data())
        if not self._is_kernelized():
            self._w = CArray(classifier.coef_)
        else:
            sv_idx = CArray(classifier.support_).ravel()
            self._alpha__ = CArray(classifier.dual_coef_)
            self._alpha[sv_idx] = CArray(classifier.dual_coef_)
            # we need to sort indices too to be consistent with alpha
            self._sv_idx = sv_idx.sort()
        self._b = CArray(classifier.intercept_[0])[0]

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
        v = self.w if not self._is_kernelized() else self.alpha
        score = CArray(x.dot(v.T)).todense() + self.b
        if self.n_classes > 2:  # return current score matrix
            scores = score
        else:  # concatenate scores
            scores = CArray.ones(shape=(x.shape[0], self.n_classes))
            scores[:, 0] = -score.ravel().T
            scores[:, 1] = score.ravel().T
        return scores

    def _backward(self, w):
        v = self.w if not self._is_kernelized() else self.alpha
        if self.n_classes > 2:
            return w.dot(v)
        else:
            return w[0] * -v + w[1] * v

    #  --------------- OTHER GRADIENTS ----------------

    def _sv_margin(self, tol=1e-6):
        """Return the margin support vectors."""
        if self.n_classes > 2:
            raise ValueError("SVM is not binary!")

        assert (self.kernel.rv.shape[0] == self.alpha.shape[1])

        alpha = self.alpha.todense()
        s = alpha.find(
            (abs(alpha) >= tol) *
            (abs(alpha) <= self.C - tol))
        if len(s) > 0:
            return self.kernel.rv[s, :], CArray(s)
        else:  # no margin SVs
            return None, None

    def hessian_tr_params(self, x=None, y=None):
        """
        Hessian of the training objective w.r.t. the classifier parameters.
        """
        xs, _ = self._sv_margin()  # these points are already normalized
        s = xs.shape[0]

        H = CArray.ones(shape=(s + 1, s + 1))
        sv = self.kernel.rv  # store and recover current sv set
        self.kernel.rv = xs
        H[:s, :s] = self.kernel._forward(xs)  # no preprocessing
        self.kernel.rv = sv
        H[-1, -1] = 0

        return H

    def grad_f_params(self, x, y=1):
        """Derivative of the decision function w.r.t. alpha and b

        Parameters
        ----------
        x : CArray
            Samples on which the training objective is computed.
        y : int
            Index of the class wrt the gradient must be computed.

        """
        xs, _ = self._sv_margin()  # these points are already preprocessed

        if xs is None:
            self.logger.debug("Warning: sv_margin is empty "
                              "(all points are error vectors).")
            return None

        s = xs.shape[0]  # margin support vector
        k = x.shape[0]

        Ksk_ext = CArray.ones(shape=(s + 1, k))

        sv = self.kernel.rv  # store and recover current sv set
        self.kernel.rv = xs
        Ksk_ext[:s, :] = self.kernel.forward(x).T  # x and xs are preprocessed
        self.kernel.rv = sv

        return convert_binary_labels(y) * Ksk_ext  # (s + 1) * k

    def grad_loss_params(self, x, y, loss=None):
        """
        Derivative of the loss w.r.t. the classifier parameters (alpha, b)

        dL / d_params = dL / df * df / d_params

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the loss is computed.
        y : CArray
            Labels of the training samples.
        loss: None (default) or CLoss
            If the loss is equal to None (default) the classifier loss is used
            to compute the derivative.

        """
        if loss is None:
            loss = self._loss

        # compute the loss derivative w.r.t. alpha
        f_params = self.grad_f_params(x)  # (s + 1) * n_samples
        scores = self.decision_function(x)
        dL_s = loss.dloss(y, score=scores).atleast_2d()
        dL_params = dL_s * f_params  # (s + 1) * n_samples
        grad = self.C * dL_params
        return grad

    def grad_tr_params(self, x, y):
        """Derivative of the classifier training objective w.r.t.
        the classifier parameters.

        dL / d_params = dL / df * df / d_params + dReg / d_params

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the loss is computed.
        y : CArray
            Features of the training samples

        """
        grad = self.grad_loss_params(x, y)  # (s+1) * n_samples

        # compute the regularizer derivative w.r.t alpha
        xs, idx = self._sv_margin()

        sv = self.kernel.rv  # store and recover current sv set
        self.kernel.rv = xs
        K = self.kernel._forward(xs)
        self.kernel.rv = sv

        d_reg = 2 * K.dot(self.alpha[idx].T)  # s * 1

        # add the regularizer to the gradient of the alphas
        s = idx.size
        grad[:s, :] += d_reg
        return grad  # (s+1) * n_samples
