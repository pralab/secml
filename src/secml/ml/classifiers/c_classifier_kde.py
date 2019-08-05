"""
.. module:: ClassifierKernelDensityEstimator
   :synopsis: Kernel Density Estimator

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray
from secml.ml.classifiers import CClassifier
from secml.ml.classifiers.clf_utils import \
    check_binary_labels
from secml.ml.kernel import CKernel
from secml.utils.mixed_utils import check_is_fitted
from secml.ml.classifiers.gradients import CClassifierGradientKDEMixin


# TODO: extend to multiclass, use SkLearn!
class CClassifierKDE(CClassifier, CClassifierGradientKDEMixin):
    """Kernel Density Estimator
    
    Parameters
    ----------
    kernel : None or CKernel subclass, optional
        Instance of a CKernel subclass to be used for computing
        similarity between patterns. If None (default), a linear
        SVM will be created.
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'kde'

    See Also
    --------
    CKernel : Pairwise kernels and metrics.

    """
    __class_type = 'kde'

    def __init__(self, kernel=None, preprocess=None):

        # Calling CClassifier init
        super(CClassifierKDE, self).__init__(preprocess=preprocess)

        # Setting up the kernel function
        kernel_type = 'linear' if kernel is None else kernel
        self._kernel = CKernel.create(kernel_type)

        self._training_samples = None  # slot store training samples

    def is_linear(self):
        """Return True if the classifier is linear."""
        if (self.preprocess is None or self.preprocess is not None and
            self.preprocess.is_linear()) and self.is_kernel_linear():
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
        check_is_fitted(self, 'training_samples')
        super(CClassifierKDE, self)._check_is_fitted()

    @property
    def kernel(self):
        """Kernel function (None if a linear classifier)."""
        return self._kernel

    @property
    def training_samples(self):
        return self._training_samples

    @training_samples.setter
    def training_samples(self, value):
        self._training_samples = value

    def _fit(self, dataset):
        """Trains the One-Vs-All Kernel Density Estimator classifier.

        The following is a private method computing one single
        binary (2-classes) classifier of the OVA schema.

        Representation of each classifier attribute for the multiclass
        case is explained in corresponding property description.

        Parameters
        ----------
        dataset : CDataset
            Binary (2-class) training set. Must be a :class:`.CDataset`
            instance with patterns data and corresponding labels.

        Returns
        -------
        trained_cls : CClassifierKDE
            Instance of the KDE classifier trained using input dataset.

        """
        if dataset.num_classes > 2:
            raise ValueError("training can be performed on (1-classes) "
                             "or binary datasets only. If dataset is binary "
                             "only negative class are considered.")

        negative_samples_idx = dataset.Y.find(dataset.Y == 0)

        if negative_samples_idx is None:
            raise ValueError("training set must contain same negative samples")

        self._training_samples = dataset.X[negative_samples_idx, :]

        self.logger.info("Number of training samples: {:}"
                         "".format(self._training_samples.shape[0]))

        return self

    def _decision_function(self, x, y=None):
        """Computes the decision function for each pattern in x.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : {0, 1}, optional
            The label of the class wrt the function should be calculated.
            Default is 1.

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        check_binary_labels(y)  # Label should be in {0, 1}

        scores = CArray.ones(shape=(x.shape[0], self.n_classes))
        k = self.kernel.k(x, self._training_samples)
        scores[:, 0] = CArray(k).mean(keepdims=False, axis=1).T
        scores[:, 1] = 1 - scores[:, 0]

        if y is not None:
            return scores[:, y].ravel()
        else:
            return scores
