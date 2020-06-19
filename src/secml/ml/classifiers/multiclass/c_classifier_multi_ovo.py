"""
.. module:: CClassifierMulticlassOVO
   :synopsis: One-Vs-One multiclass classifier

.. moduleauthor:: Giorgio Piras <giorgiopiras4@gmail.com>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from itertools import combinations

from secml.ml.classifiers.multiclass import CClassifierMulticlass
from secml.ml.classifiers.gradients import CClassifierGradientMixin
from secml.array import CArray
from secml.data import CDataset
from secml.parallel import parfor2


def _fit_one_ovo(bin_clf_idx, multi_ovo, dataset, verbose):
    """Fit the OVO classifier given an index.

    This method fits a one-vs-one classifier wrt the
    positive and negative labels taken from the list
    clf_pair_idx at the index bin_clf_idx.

    Parameters
    ----------
    bin_clf_idx : int
        Index of the binary classifier
    multi_ovo : CClassifierMulticlassOVO
        Instance of the multiclass OVO classifier.
    dataset : CDataset
        Training set. Must be a :class:`.CDataset` instance with
        patterns data and corresponding labels.
    verbose : int
        Verbosity level of the logger.

    """
    # Resetting verbosity level. This is needed as objects
    # change id  when passed to subprocesses and our logging
    # level is stored per-object looking to id
    multi_ovo.verbose = verbose

    # Take the classes indices
    tr_class_idx = multi_ovo._clf_pair_idx[bin_clf_idx][0]
    vs_class_idx = multi_ovo._clf_pair_idx[bin_clf_idx][1]

    multi_ovo.logger.info(
        "Training class {:} against class: {:}".format(
            tr_class_idx, vs_class_idx))

    # Create the training dataset
    train_ds = multi_ovo.binarize_subset(tr_class_idx, vs_class_idx, dataset)

    # Extracting the internal classifier
    classifier_instance = multi_ovo._binary_classifiers[bin_clf_idx]
    # Setting verbosity level
    classifier_instance.verbose = multi_ovo.verbose
    # Training the one-vs-ne classifier
    classifier_instance.fit(train_ds.X, train_ds.Y)

    return classifier_instance


def _forward_one_ovo(clf_idx, multi_ovo, test_x, verbose):
    """Perform forward on an OVO classifier.

    Parameters
    ----------
    clf_idx : int
        Index of the OVO classifier.
    multi_ovo : CClassifierMulticlassOVO
        Instance of the multiclass OVO classifier.
    test_x : CArray
        Test data as 2D CArray.
    verbose : int
        Verbosity level of the logger.

    """
    # Resetting verbosity level. This is needed as objects
    # change id  when passed to subprocesses and our logging
    # level is stored per-object looking to id
    multi_ovo.verbose = verbose

    multi_ovo.logger.info(
        "Forward for classes: {:}".format(multi_ovo._clf_pair_idx[clf_idx]))

    # Perform forward on data for current class classifier
    return multi_ovo._binary_classifiers[clf_idx].forward(test_x)


class CClassifierMulticlassOVO(CClassifierMulticlass,
                               CClassifierGradientMixin):
    """OVO (One-Vs-One) Multiclass Classifier.

    Parameters
    ----------
    classifier : unbound class
        Unbound (not initialized) CClassifier subclass.
    kwargs : any
        Any other construction parameter for each OVA classifier.

    Attributes
    ----------
    class_type : 'ovo'

    """
    __class_type = 'ovo'

    def __init__(self, classifier, preprocess=None, **clf_params):

        super(CClassifierMulticlassOVO, self).__init__(
            classifier=classifier,
            preprocess=preprocess,
            **clf_params
        )

        # List with the binary classifiers classes pairs
        self._clf_pair_idx = None

    @property
    def clf_pair_idx(self):
        """List with the binary classifiers' classes (indices) pairs."""
        return self._clf_pair_idx

    def _fit(self, x, y):
        """Trains the classifier.

        All the One-Vs-One classifier are trained for each dataset class.

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features).
        y : CArray
            Array of shape (n_samples,) containing the class labels.

        Returns
        -------
        trained_cls : CClassifierMulticlassOVO
            Instance of the classifier trained using input dataset.

        """
        # Number of unique classes
        n_classes = y.unique().size
        # Number of classifiers to be trained
        ovo_clf_number = int((n_classes * (n_classes - 1)) / 2)
        # Preparing the binary classifiers
        self.prepare(ovo_clf_number)
        # Preparing the list of binary classifiers indices
        self._clf_pair_idx = list(combinations(range(n_classes), 2))

        # Fit a one-vs-one classifier
        # Use the specified number of workers
        self._binary_classifiers = parfor2(_fit_one_ovo,
                                           self.num_classifiers,
                                           self.n_jobs, self, CDataset(x, y),
                                           self.verbose)

        return self

    @staticmethod
    def binarize_subset(tr_class_idx, vs_class_idx, dataset):
        """Returns the binary dataset tr_class_idx vs vs_class_idx.

        Parameters
        ----------
        tr_class_idx : int
            Index of the target class.
        vs_class_idx: int
            Index of the opposing class.
        dataset : CDataset
            Dataset from which the subset should be extracted.

        Returns
        -------
        bin_subset : CDataset
            Binarized subset.

        """
        tr_class = dataset.classes[tr_class_idx]
        vs_class = dataset.classes[vs_class_idx]

        tr_idx = dataset.Y.find(dataset.Y == tr_class)
        vs_idx = dataset.Y.find(dataset.Y == vs_class)

        subset = dataset[tr_idx + vs_idx, :]

        # Using get_labels_ovr to avoid redundant functions
        return CDataset(
            subset.X, subset.get_labels_ovr(tr_class), header=dataset.header)

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
        raise NotImplementedError

    def _forward(self, x):
        """Computes the decision function for each pattern in x.

        To evaluate correctly, scores are also taken from the
        negative classes in each binary classifier.


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
        scores = CArray.zeros(shape=(x.shape[0], self.n_classes))

        # Discriminant function is now called for each different class
        res = parfor2(_forward_one_ovo,
                      self.num_classifiers,
                      self.n_jobs, self, x,
                      self.verbose)

        # Building results array
        for i in range(self.num_classifiers):
            # Adjusting the scores for the OVO scheme
            idx0 = self._clf_pair_idx[i][0]
            idx1 = self._clf_pair_idx[i][1]
            scores[:, idx0] += res[i][:, 1]
            scores[:, idx1] += res[i][:, 0]

        return scores / (self.n_classes - 1)

    def _backward(self, w):
        """Implement gradient of decision function wrt x."""
        if w is None:
            raise ValueError('Pre-multiplying vector w cannot be None.')

        grad = None  # To accumulate grads
        for i in range(self.num_classifiers):  # TODO parfor

            # Taking the scores
            idx0 = self._clf_pair_idx[i][0]
            idx1 = self._clf_pair_idx[i][1]

            w_pos = CArray([1, 0])
            grad_pos = w[idx0] * \
                self._binary_classifiers[i].gradient(self._cached_x, w_pos)

            w_neg = CArray([0, 1])
            grad_neg = w[idx1] * \
                self._binary_classifiers[i].gradient(self._cached_x, w_neg)

            # Adjusting the scores for the OVO scheme
            grad = grad_pos if grad is None else grad + grad_pos
            grad += grad_neg

        # A trade-off between the two classes
        return -grad / (self.n_classes - 1)
