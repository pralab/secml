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

        # Classifier parameters
        self.C = C

        # After-training attributes
        self._w = None
        self._b = None

        self._kernel = CKernel.create(kernel)

    def _is_kernel_linear(self):
        if self._kernel.__class__ == 'linear':
            return True
        else:
            return False

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b

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
        # Setting up classifier parameters
        if self._is_kernel_linear():
            kernel = 'linear'
            z = x
            self._w = CArray.zeros(shape=(self.n_classes, self.n_features))
        else:
            kernel = 'precomputed'
            z = self._kernel.k(x)
            self._alpha = CArray.zeros(shape=(self.n_classes, z.shape[0]))
        self._b = CArray.zeros(shape=(self.n_classes,))

        # ova
        for k, c in enumerate(self.classes):
            # Training one vs all
            classifier = SVC(C=self.C, kernel=kernel)
            labels = y == c
            classifier.fit(z.get_data(), labels.tondarray())
            if self._is_kernel_linear():
                self._w[k, :] = CArray(classifier.coef_.ravel())
            else:
                self._sv_idx = CArray(classifier.support_).ravel()
                print(classifier.n_support_, classifier.support_.shape)
                self._alpha[k, self._sv_idx] = CArray(classifier.dual_coef_)
            self._b[k] = CArray(classifier.intercept_[0])[0]

        return self

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
            (n_patterns, n_features).

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_samples,) if `y` is not None,
            otherwise a (n_samples, n_classes) array.

        """
        if self._is_kernel_linear():
            scores = CArray(x.dot(self.w.T)) + self.b
        else:
            print(self._kernel.rv.shape)
            z = self._kernel.forward(x)
            print(z.shape, self._alpha.shape)
            scores = CArray(z.dot(self._alpha.T)) + self.b
        return scores
