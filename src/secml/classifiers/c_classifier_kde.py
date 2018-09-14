"""
.. module:: CClassifierKernelDensityEstimator
   :synopsis: Kernel Density Estimator

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml.array import CArray
from secml.classifiers import CClassifier
from secml.kernel import CKernel


class CClassifierKDE(CClassifier):
    """Kernel Density Estimator
    
    Parameters
    ----------
    kernel : None or CKernel subclass, optional
        Instance of a CKernel subclass to be used for computing
        similarity between patterns. If None (default), a linear
        SVM will be created.

    See Also
    --------
    .CKernel : Pairwise kernels and metrics.

    """
    class_type = 'kde'

    def __init__(self, kernel=None, normalizer=None):

        # Calling CClassifier init
        super(CClassifierKDE, self).__init__(normalizer=normalizer)

        # After-training attributes
        self._training_samples = None  # slot store training samples

        # Setting up the kernel function
        kernel_type = 'linear' if kernel is None else kernel
        self._kernel = CKernel.create(kernel_type)

    def __clear(self):
        self._training_samples = None

    def is_clear(self):
        """Returns True if object is clear."""
        return self._training_samples is None and \
            super(CClassifierKDE, self).is_clear()

    def is_linear(self):
        """Return True if the classifier is linear."""
        if (self.normalizer is None or self.normalizer is not None and
                self.normalizer.is_linear()) and self.is_kernel_linear():
            return True
        return False

    def is_kernel_linear(self):
        """Return True if the kernel is None or linear."""
        if self.kernel is None or self.kernel.class_type == 'linear':
            return True
        return False

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

    def _train(self, dataset):
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

    def _discriminant_function(self, x, label):
        """Compute the probability of samples to being in class with specified label 

        Parameters
        ----------
        x : CArray or array_like
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        label : int
            The label of the class with respect to which the function
            should be calculated.

        Returns
        -------
        score : CArray or scalar
            Probability for patterns to be in class with specified label

        """
        if label == 1:
            return 1 - CArray(
                self.kernel.k(x, self._training_samples)).mean(keepdims=False)
        else:
            return CArray(
                self.kernel.k(x, self._training_samples)).mean(keepdims=False)

    def _gradient_x(self, x, y=1):
        """Computes the gradient of the linear classifier's discriminant function wrt 'x'.

        For the linear classifier this is equal to simply
        return the weights vector w.

        The input point x can be in fact ignored.

        Returns
        -------
        grad : CArray or scalar
            The gradient of the linear classifier's decision function.
            This is equal to the vector with each feature's weight.
            Format (dense or sparse) depends on training data.
        y : int, optional
            Index of the class wrt the gradient must be computed.
            Default is 1, corresponding to the positive class.

        """
        k = self.kernel.gradient(self._training_samples, x)
        return - (2 * y - 1) * k.mean(axis=0)
