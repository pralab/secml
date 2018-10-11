"""
.. module:: ClassifierKernelDensityEstimator
   :synopsis: Kernel Density Estimator

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.array import CArray
from secml.classifiers import CClassifier
from secml.classifiers.clf_utils import check_binary_labels, convert_binary_labels
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

    def discriminant_function(self, x, y=1):
        """Computes the discriminant function for each pattern in x.

        If a normalizer has been specified, input is normalized
         before computing the discriminant function.

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
            Value of the discriminant function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        if self.is_clear():
            raise ValueError("make sure the classifier is trained first.")

        x = x.atleast_2d()  # Ensuring input is 2-D

        # Normalizing data if a normalizer is defined
        if self.normalizer is not None:
            x = self.normalizer.normalize(x)

        return self._discriminant_function(x, y=y)

    def _discriminant_function(self, x, y=1):
        """Computes the discriminant function for each pattern in x.

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
            Value of the discriminant function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        check_binary_labels(y)  # Label should be in {0, 1}

        x = x.atleast_2d()  # Ensuring input is 2-D

        if y == 1:
            return 1 - CArray(self.kernel.k(x, self._training_samples)).mean(
                                                        keepdims=False, axis=1)
        else:
            return CArray(self.kernel.k(x, self._training_samples)).mean(
                                                        keepdims=False, axis=1)

    def _gradient_f(self, x, y=1):
        """Computes the gradient of the KDE classifier's decision function
         wrt decision function input.

        Returns
        -------
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
        k = self.kernel.gradient(self._training_samples, x)
        # Gradient sign depends on input label (0/1)
        return - convert_binary_labels(y) * k.mean(axis=0)
