"""
.. module:: ClassifierMulticlassOVA
   :synopsis: One-Vs-All Multiclass Classifier

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.classifiers.multiclass import CClassifierMulticlass
from secml.array import CArray
from secml.data import CDataset
from secml.parallel import parfor2


def _train_one_ova(
        tr_class_idx, multi_ova, dataset, verbose):
    """Train a OVA classifier.

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

    # Extracting one-vs-all classifier
    classifier_instance = multi_ova.binary_classifiers[tr_class_idx]
    # Setting verbosity level
    classifier_instance.verbose = multi_ova.verbose
    # Training one-vs-all classifier
    classifier_instance.train(train_ds)

    return classifier_instance


class CClassifierMulticlassOVA(CClassifierMulticlass):
    """OVA (One-Vs-All) Multiclass Classifier.

    Parameters
    ----------
    classifier : unbound class
        Unbound (not initialized) CClassifier subclass.
    kwargs : any
        Any other construction parameter for each OVA classifier.

    """
    class_type = 'ova'

    def _train(self, dataset, n_jobs=1):
        """Trains the classifier.

        A One-Vs-All classifier is trained for each dataset class.

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
        trained_cls : CClassifierMulticlassOVA
            Instance of the classifier trained using input dataset.

        """
        # Preparing the binary classifiers
        self.prepare(dataset.num_classes)

        # Train a one-vs-all classifier for each class
        # Use the specified number of workers
        self._binary_classifiers = parfor2(_train_one_ova,
                                           self.classes.size,
                                           n_jobs, self, dataset,
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
            dataset.X, dataset.get_labels_asbinary(dataset.classes[class_idx]))

    def _discriminant_function(self, x, label):
        """Compute the discriminant function against class `label`.

        For OVA, this is the output of the label^th classifier.

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
            Flat array of shape (n_patterns,) with discriminant function
            value of each test pattern or scalar if n_patterns == 1.

        """
        self.logger.info(
            "Getting discriminant function against class: {:}".format(label))
        # Getting predicted score for current class classifier
        return self.binary_classifiers[label].discriminant_function(x, label=1)
