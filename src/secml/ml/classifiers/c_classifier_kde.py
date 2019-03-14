"""
.. module:: ClassifierKernelDensityEstimator
   :synopsis: Kernel Density Estimator

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.array import CArray
from secml.ml.classifiers import CClassifier
from secml.ml.classifiers.clf_utils import \
    check_binary_labels, convert_binary_labels
from secml.ml.kernel import CKernel
from secml.utils.mixed_utils import check_is_fitted


class CClassifierKDE(CClassifier):
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
    .CKernel : Pairwise kernels and metrics.

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

    def decision_function(self, x, y=1):
        """Computes the decision function for each pattern in x.

        If a preprocess has been specified, input is normalized
         before computing the decision function.

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
        self._check_is_fitted()

        x = x.atleast_2d()  # Ensuring input is 2-D

        # Transform data if a preprocess is defined
        x = self._preprocess_data(x)

        return self._decision_function(x, y=y)

    def _decision_function(self, x, y=1):
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

        x = x.atleast_2d()  # Ensuring input is 2-D

        if y == 1:
            return 1 - CArray(self.kernel.k(x, self._training_samples)).mean(
                                                        keepdims=False, axis=1)
        else:
            return CArray(self.kernel.k(x, self._training_samples)).mean(
                                                        keepdims=False, axis=1)

    def gradient_f_x(self, x, y=1, **kwargs):
        """Computes the gradient of the classifier's output wrt input.

        Parameters
        ----------
        x : CArray
            The gradient is computed in the neighborhood of x.
        y : int, optional
            Index of the class wrt the gradient must be computed. Default 1.
        **kwargs
            Optional parameters for the function that computes the
            gradient of the decision function. See the description of
            each classifier for a complete list of optional parameters.

        Returns
        -------
        gradient : CArray
            Gradient of the classifier's output wrt input. Vector-like array.

        """
        return super(CClassifierKDE, self).gradient_f_x(x, y, **kwargs)

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
        return - convert_binary_labels(y) * k.mean(axis=0, keepdims=False)
