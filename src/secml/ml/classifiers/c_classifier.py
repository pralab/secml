"""
.. module:: CClassifier
   :synopsis: Interface and common functions for classification

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod, abstractproperty

from secml.core import CCreator
from secml.array import CArray
from secml.data import CDataset
from secml.ml.features.normalization import CNormalizer
from secml.parallel import parfor2


def _classify_one(tr_class_idx, clf, test_x, verbose):
    """Performs classification wrt class of label `tr_class_idx`.

    Parameters
    ----------
    tr_class_idx : int
        Index of the label against which the classifier should be trained.
    clf : CClassifier
        Instance of the classifier.
    test_x : CArray
        Test data as 2D CArray.
    verbose : int
        Verbosity level of the logger.

    """
    # Resetting verbosity level. This is needed as objects
    # change id  when passed to subprocesses and our logging
    # level is stored per-object looking to id
    clf.verbose = verbose
    # Getting predicted data for current class classifier
    return clf.decision_function(test_x, y=tr_class_idx)


class CClassifier(CCreator):
    """Abstract class that defines basic methods for Classifiers.

    A classifier assign a label (class) to new patterns using the
    informations learned from training set.

    This interface implements a set of generic methods for training
    and classification that can be used for every algorithms. However,
    all of them can be reimplemented if specific routines are needed.

    Parameters
    ----------
    normalizer : str, CNormalizer
        Features normalizer to applied to input data.
        Can be a CNormalizer subclass or a string with the desired
        normalizer type. If None, input data is used as is.

    """
    __metaclass__ = ABCMeta
    __super__ = 'CClassifier'

    def __init__(self, normalizer=None):
        # List of classes on which training has been performed
        self._classes = None
        # Number of features of the training dataset
        self._n_features = None
        # Data normalizer
        # Setting up the kernel function
        self.normalizer = normalizer if normalizer is None \
            else CNormalizer.create(normalizer)

    def __clear(self):
        """Reset the object."""
        self._classes = None
        self._n_features = None
        if self.normalizer is not None:
            self.normalizer.clear()

    def __is_clear(self):
        """Returns True if object is clear."""
        if self._classes is not None:
            return False
        if self._n_features is not None:
            return False
        if self.normalizer is not None and not self.normalizer.is_clear():
            return False
        return True

    @property
    def classes(self):
        """Return the list of classes on which training has been performed."""
        return self._classes

    @property
    def n_classes(self):
        """Number of classes of training dataset."""
        return self.classes.size if self.classes is not None else None

    @property
    def n_features(self):
        """Number of features"""
        return self._n_features

    def is_linear(self):
        """Return true for linear classifiers, false otherwise"""
        return False

    @abstractmethod
    def _fit(self, dataset):
        """Private method that trains the One-Vs-All classifier.
        Must be reimplemented by subclasses.

        Parameters
        ----------
        dataset : CDataset
            Training set. Must be a :class:`.CDataset` instance with
            patterns data and corresponding labels.

        Returns
        -------
        trained_cls : CClassifier
            Instance of the classifier trained using input dataset.

        """
        raise NotImplementedError()

    def fit(self, dataset, n_jobs=1):
        """Trains the classifier.

        If a normalizer has been specified,
        input is normalized before training.

        For multiclass case see `.CClassifierMulticlass`.

        Parameters
        ----------
        dataset : CDataset
            Training set. Must be a :class:`.CDataset` instance with
            patterns data and corresponding labels.
        n_jobs : int
            Number of parallel workers to use for training the classifier.
            Default 1. Cannot be higher than processor's number of cores.

        Returns
        -------
        trained_cls : CClassifier
            Instance of the classifier trained using input dataset.

        """
        if not isinstance(dataset, CDataset):
            raise TypeError(
                "training set should be provided as a CDataset object.")

        # Resetting the classifier
        self.clear()

        # Storing dataset classes
        self._classes = dataset.classes
        self._n_features = dataset.num_features

        data_x = dataset.X
        # Normalizing data if a normalizer is defined
        if self.normalizer is not None:
            data_x = self.normalizer.fit_normalize(dataset.X)

        # Data is ready: fit the classifier
        try:  # Try to use parallelization
            self._fit(CDataset(data_x, dataset.Y), n_jobs=n_jobs)
        except TypeError:  # Parallelization is probably not supported
            self._fit(CDataset(data_x, dataset.Y))

        return self

    @abstractmethod
    def _decision_function(self, x, y):
        """Private method that computes the decision function.

        .. warning:: Must be reimplemented by a subclass of `.CClassifier`.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : int
            The label of the class wrt the function should be calculated.

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        raise NotImplementedError()

    def decision_function(self, x, y):
        """Computes the decision function for each pattern in x.

        If a normalizer has been specified, input is normalized
        before computing the decision function.

        .. note::

            The actual decision function should be implemented
            case by case inside :meth:`_decision_function` method.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : int
            The label of the class wrt the function should be calculated.

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_patterns,).

        Warnings
        --------
        This method implements a generic formulation where the
         decision function is computed separately for each pattern.
         It's convenient to override this when the function can be computed
         for all patterns at once to improve performance.

        """
        if self.is_clear():
            raise ValueError("make sure the classifier is trained first.")

        x = x.atleast_2d()  # Ensuring input is 2-D

        # Normalizing data if a normalizer is defined
        if self.normalizer is not None:
            x = self.normalizer.normalize(x)

        score = CArray.ones(shape=x.shape[0])
        for i in xrange(x.shape[0]):
            score[i] = self._decision_function(x[i, :], y)

        return score

    def predict(self, x, return_decision_function=False, n_jobs=1):
        """Perform classification of each pattern in x.

        If a normalizer has been specified,
         input is normalized before classification.

        Parameters
        ----------
        return_decision_function
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        return_decision_function : bool, optional
            Whether to return the decision_function value along
            with predictions. Default False.
        n_jobs : int, optional
            Number of parallel workers to use for classification.
            Default 1. Cannot be higher than processor's number of cores.

        Returns
        -------
        labels : CArray
            Flat dense array of shape (n_patterns,) with the label assigned
             to each test pattern. The classification label is the label of
             the class associated with the highest score.
        scores : CArray, optional
            Array of shape (n_patterns, n_classes) with classification
             score of each test pattern with respect to each training class.
            Will be returned only if `return_decision_function` is True.

        Warnings
        --------
        This method implements a generic formulation where the
         classification score is computed separately for training class.
         It's convenient to override this when the score can be computed
         for one of the classes only, e.g. for binary classifiers the score
         for the positive/negative class is commonly the negative of the
         score of the other class.

        """
        x = x.atleast_2d()  # Ensuring input is 2-D

        scores = CArray.ones(shape=(x.shape[0], self.n_classes))

        # Compute the decision function for each training class in parallel
        res = parfor2(_classify_one, self.n_classes,
                      n_jobs, self, x, self.verbose)

        # Build results array by extracting the scores for each training class
        for i in xrange(self.n_classes):
            scores[:, i] = CArray(res[i]).T

        # The classification label is the label of the class
        # associated with the highest score
        labels = scores.argmax(axis=1).ravel()

        return (labels, scores) if return_decision_function is True else labels

    def estimate_parameters(self, dataset, parameters, splitter, metric,
                            pick='first', perf_evaluator='xval', n_jobs=1):
        """Estimate parameter that give better result respect a chose metric.

        Parameters
        ----------
        dataset : CDataset
            Dataset to be used for evaluating parameters.
        parameters : dict
            Dictionary with each entry as {parameter: list of values to test}.
            Example: {'C': [1, 10, 100],
                      'gamma': list(10.0 ** CArray.arange(-4, 4))}
        splitter : CDataSplitter or str
            Object to use for splitting the dataset into train and validation.
            A splitter type can be passed as string, in this case all
            default parameters will be used. For data splitters, num_folds
            is set to 3 by default.
            See CDataSplitter docs for more informations.
        metric : CMetric or str
            Object with the metric to use while evaluating the performance.
            A metric type can be passed as string, in this case all
            default parameters will be used.
            See CMetric docs for more informations.
        pick : {'first', 'last', 'random'}, optional
            Defines which of the best parameters set pick.
            Usually, 'first' correspond to the smallest parameters while
            'last' correspond to the biggest. The order is consistent
            to the parameters dict passed as input.
        perf_evaluator : CPerfEvaluator or str, optional
            Performance Evaluator to use. Default 'xval'.
        n_jobs : int, optional
            Number of parallel workers to use for performance evaluation.
            Default 1. Cannot be higher than processor's number of cores.

        Returns
        -------
        best_parameters : dict
            Dictionary of best parameters found through performance evaluation.

        """
        # Import here as is only needed if this function is called
        from secml.ml.peval import CPerfEvaluator

        # Initialize the evaluator
        perf_eval = CPerfEvaluator.create(perf_evaluator, splitter, metric)
        # Set verbosity level to be the same of classifier
        # Classifier verbosity will be set to 0 wile estimating params
        perf_eval.verbose = self.verbose

        # Evaluate the best parameters for the classifier (self)
        best_params = perf_eval.evaluate_params(
            self, dataset, parameters, pick=pick, n_jobs=n_jobs)[0]

        # Clear estimator as parameters are going to change
        self.clear()

        # Set the best parameters in classifier
        self.set_params(best_params)

        return best_params

    def gradient_f_x(self, x, **kwargs):
        """Computes the gradient of the classifier's output wrt input.

        Parameters
        ----------
        x : CArray
            The gradient is computed in the neighborhood of x.
        **kwargs
            Optional parameters for the function that computes the
            gradient of the decision function. See the description of
            each classifier for a complete list of optional parameters.

        Returns
        -------
        gradient : CArray
            Gradient of the classifier's output wrt input. Vector-like array.

        """
        # Get the derivative of the normalizer (if defined)
        if self.normalizer is not None:
            grad_norm = self._gradient_n(x)
            # Normalize data before compute the classifier gradient
            x = self.normalizer.normalize(x)
        else:  # Normalizer not defined, use a "neutral" value
            grad_norm = CArray([1])

        # Get the derivative of decision_function
        try:
            grad_f = self._gradient_f(x, **kwargs)
        except NotImplementedError:
            raise NotImplementedError("{:} does not implement `gradient_f_x`"
                                      "".format(self.__class__.__name__))

        return (grad_f * grad_norm).ravel()

    def gradient_loss_params(self, **kwargs):
        """Computed the gradient of the classifier's loss wrt train parameters.

        Parameters
        ----------
        **kwargs
            Optional parameters. See the description of each
            classifier for a complete list of optional parameters.

        Returns
        -------
        gradient : CArray
            Gradient of the classifier's loss wrt train parameters.

        """
        raise NotImplementedError(
            "{:} does not implement `gradient_loss_params`"
            "".format(self.__class__.__name__))

    def _gradient_f(self, x, y):
        """Computes the gradient of the classifier's decision function
         wrt decision function input.

        Parameters
        ----------
        x : CArray
            The gradient is computed in the neighborhood of x.
        y : int
            Index of the class wrt the gradient must be computed.

        Returns
        -------
        gradient : CArray
            Gradient of the classifier's df wrt its input. Vector-like array.

        """
        raise NotImplementedError

    def _gradient_n(self, x):
        """Computes the gradient of the normalizer wrt normalizer input.

        Parameters
        ----------
        x : CArray
            The gradient is computed in the neighborhood of x.

        Returns
        -------
        gradient : CArray
            Gradient of the normalizer wrt its input. Vector-like array.

        """
        grad_norm = self.normalizer.gradient(x)
        if self.normalizer.is_linear():
            # Gradient of linear normalizers is an 'I * w' array
            # To avoid returning all the zeros, extract the main diagonal
            grad_norm = grad_norm.diag()
        return grad_norm.ravel()
