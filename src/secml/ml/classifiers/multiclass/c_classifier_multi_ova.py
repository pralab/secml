"""
.. module:: CClassifierMulticlassOVA
   :synopsis: One-Vs-All multiclass classifier

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.ml.classifiers.multiclass import CClassifierMulticlass
from secml.ml.classifiers.gradients import CClassifierGradientMixin
from secml.array import CArray
from secml.data import CDataset
from secml.parallel import parfor2


def _fit_one_ova(
        tr_class_idx, multi_ova, dataset, verbose):
    """Fit a OVA classifier.

    Parameters
    ----------
    tr_class_idx : int
        Index of the label against which the classifier should be trained.
    multi_ova : CClassifierMulticlassOVA
        Instance of the multiclass OVA classifier.
    dataset : CDataset
        Training set. Must be a :class:`.CDataset` instance with
        patterns data and corresponding labels.
    verbose : int
        Verbosity level of the logger.

    """
    # Resetting verbosity level. This is needed as objects
    # change id  when passed to subprocesses and our logging
    # level is stored per-object looking to id
    multi_ova.verbose = verbose

    multi_ova.logger.info(
        "Training against class: {:}".format(tr_class_idx))

    # Binarizing dataset
    train_ds = multi_ova.binarize_dataset(tr_class_idx, dataset)

    # Extracting the internal one-vs-all classifier
    classifier_instance = multi_ova._binary_classifiers[tr_class_idx]
    # Setting verbosity level
    classifier_instance.verbose = multi_ova.verbose
    # Training one-vs-all classifier
    classifier_instance.fit(train_ds.X, train_ds.Y)

    return classifier_instance


def _forward_one_ova(tr_class_idx, multi_ova, test_x, verbose):
    """Perform forward on an OVA classifier.

    Parameters
    ----------
    tr_class_idx : int
        Index of the OVA classifier.
    multi_ova : CClassifierMulticlassOVA
        Instance of the multiclass OVA classifier.
    test_x : CArray
        Test data as 2D CArray.
    verbose : int
        Verbosity level of the logger.

    """
    # Resetting verbosity level. This is needed as objects
    # change id  when passed to subprocesses and our logging
    # level is stored per-object looking to id
    multi_ova.verbose = verbose

    multi_ova.logger.info(
        "Forward for class: {:}".format(tr_class_idx))

    # Perform forward on data for current class classifier
    return multi_ova._binary_classifiers[tr_class_idx].forward(test_x)[:, 1]


class CClassifierMulticlassOVA(CClassifierMulticlass,
                               CClassifierGradientMixin):
    """OVA (One-Vs-All) Multiclass Classifier.

    Parameters
    ----------
    classifier : unbound class
        Unbound (not initialized) CClassifier subclass.
    kwargs : any
        Any other construction parameter for each OVA classifier.

    Attributes
    ----------
    class_type : 'ova'

    """
    __class_type = 'ova'

    def __init__(self, classifier, preprocess=None, n_jobs=1, **clf_params):

        super(CClassifierMulticlassOVA, self).__init__(
            classifier=classifier,
            preprocess=preprocess,
            n_jobs=n_jobs,
            **clf_params
        )

    def _fit(self, x, y):
        """Trains the classifier.

        A One-Vs-All classifier is trained for each dataset class.

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features).
        y : CArray
            Array of shape (n_samples,) containing the class labels.

        Returns
        -------
        trained_cls : CClassifierMulticlassOVA
            Instance of the classifier trained using input dataset.

        """
        # Preparing the binary classifiers
        self.prepare(y.unique().size)

        # Fit a one-vs-all classifier for each class
        # Use the specified number of workers
        self._binary_classifiers = parfor2(_fit_one_ova,
                                           self.classes.size,
                                           self.n_jobs, self, CDataset(x, y),
                                           self.verbose)

        return self

    @staticmethod
    def binarize_dataset(class_idx, dataset):
        """Returns the dataset needed by the class_idx binary classifier.

        Parameters
        ----------
        class_idx : int
            Index of the target class.
        dataset : CDataset
            Dataset to binarize.

        Returns
        -------
        bin_dataset : CDataset
            Binarized dataset.

        """
        return CDataset(
            dataset.X, dataset.get_labels_ovr(dataset.classes[class_idx]),
            header=dataset.header)

    def _forward(self, x):
        """Computes the decision function for each pattern in x.

        For One-Vs-All (OVA) multiclass scheme,
         this is the output of the `label`^th classifier.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_samples,) if y is not None,
            otherwise a (n_samples, n_classes) array.

        """
        # Getting predicted scores for classifier associated with y
        scores = CArray.empty(shape=(x.shape[0], self.n_classes))

        # Discriminant function is now called for each different class
        res = parfor2(_forward_one_ova,
                      self.n_classes,
                      self.n_jobs, self, x,
                      self.verbose)

        # Building results array
        for i in range(self.n_classes):
            scores[:, i] = CArray(res[i])

        return scores

    def _backward(self, w):
        """Implement gradient of decision function wrt x."""
        if w is None:
            w = CArray.ones(shape=(self.n_classes,))

        # this is where we'll accumulate grads
        grad = CArray.zeros(
            shape=self._cached_x.shape, sparse=self._cached_x.issparse)

        # loop only over non-zero elements in w, to save computations
        for c in w.nnz_indices[1]:
            grad_c = self._binary_classifiers[c].grad_f_x(self._cached_x, y=1)
            grad += w[c] * grad_c
        return grad
