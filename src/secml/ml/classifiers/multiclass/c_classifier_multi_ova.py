"""
.. module:: CClassifierMulticlassOVA
   :synopsis: One-Vs-All multiclass classifier

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.ml.classifiers.multiclass import CClassifierMulticlass
from secml.ml.classifiers.multiclass.mixin_classifier_gradient_multiclass_ova import \
    CClassifierGradientMulticlassOVAMixin
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
    classifier_instance.fit(train_ds)

    return classifier_instance


class CClassifierMulticlassOVA(CClassifierMulticlass,
                               CClassifierGradientMulticlassOVAMixin):
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

    def __init__(self, classifier, preprocess=None, **clf_params):

        super(CClassifierMulticlassOVA, self).__init__(
            classifier=classifier,
            preprocess=preprocess,
            **clf_params
        )

    def _fit(self, dataset, n_jobs=1):
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

        # Fit a one-vs-all classifier for each class
        # Use the specified number of workers
        self._binary_classifiers = parfor2(_fit_one_ova,
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
            dataset.X, dataset.get_labels_ovr(dataset.classes[class_idx]),
            header=dataset.header)

    def _decision_function(self, x, y=None):
        """Computes the decision function for each pattern in x.

        For One-Vs-All (OVA) multiclass scheme,
         this is the output of the `label`^th classifier.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : int or None, optional
            The label of the class wrt the function should be calculated.
            If None, return the output for all classes.

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_samples,) if y is not None,
            otherwise a (n_samples, n_classes) array.

        """
        # Getting predicted scores for classifier associated with y
        if y is not None:
            self.logger.info(
                "Getting decision function against class: {:}".format(y))
            return self._binary_classifiers[y].decision_function(x, y=1)
        else:
            scores = CArray.ones(shape=(x.shape[0], self.n_classes))
            for i in range(self.n_classes):  # TODO parfor
                scores[:, i] = self._binary_classifiers[i].decision_function(
                    x, y=1).ravel().T
            return scores
