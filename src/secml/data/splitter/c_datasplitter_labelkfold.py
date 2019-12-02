"""
.. module:: CDataSplitterLabelKFold
   :synopsis: Label K-Fold splitting

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.array import CArray
from secml.data.splitter import CDataSplitter


class CDataSplitterLabelKFold(CDataSplitter):
    """K-Folds dataset splitting with non-overlapping labels.

    The same label will not appear in two different folds
    (the number of distinct labels has to be at least equal
    to the number of folds).

    The folds are approximately balanced in the sense that the
    number of distinct labels is approximately the same in each fold.

    Parameters
    ----------
    num_folds : int, optional
        Number of folds to create. Default 3.
        This correspond to the size of tr_idx and ts_idx lists.

    Attributes
    ----------
    class_type : 'label-kfold'

    Examples
    --------
    >>> from secml.data import CDataset
    >>> from secml.data import CDataset
    >>> from secml.data.splitter import CDataSplitterLabelKFold
    >>> ds = CDataset([[1,2],[3,4],[5,6],[7,8]], [1,0,1,2])
    >>> kfold = CDataSplitterLabelKFold(num_folds=3).compute_indices(ds)
    >>> print(kfold.num_folds)
    3
    >>> print(kfold.tr_idx)
    [CArray(2,)(dense: [1 3]), CArray(3,)(dense: [0 1 2]), CArray(3,)(dense: [0 2 3])]
    >>> print(kfold.ts_idx)
    [CArray(2,)(dense: [0 2]), CArray(1,)(dense: [3]), CArray(1,)(dense: [1])]

    """
    __class_type = 'label-kfold'

    def __init__(self, num_folds=3):

        super(CDataSplitterLabelKFold, self).__init__(num_folds=num_folds)

    def compute_indices(self, dataset):
        """Compute training set and test set indices for each fold.

        Parameters
        ----------
        dataset : CDataset
            Dataset to split.

        Returns
        -------
        CDataSplitter
            Instance of the dataset splitter with tr/ts indices.

        """
        # Resetting indices
        self._tr_idx = []
        self._ts_idx = []

        unique_labels, labels = dataset.Y.unique(return_inverse=True)
        n_labels = unique_labels.size

        if self.num_folds > n_labels:
            raise ValueError(
                    ("Cannot have number of folds ({0}) greater"
                     " than the number of classes: {1}.").format(
                        self.num_folds, n_labels))

        # Weight labels by their number of occurrences
        n_samples_per_label = labels.bincount()

        # Distribute the most frequent labels first
        indices = n_samples_per_label.argsort(axis=None)[::-1]
        n_samples_per_label = n_samples_per_label[indices]

        # Total weight of each fold
        n_samples_per_fold = CArray.zeros(self.num_folds, dtype=int)

        # Mapping from label index to fold index
        label_to_fold = CArray.zeros(n_labels, dtype=int)

        # Distribute samples by adding the largest weight to the lightest fold
        for label_index, weight in enumerate(n_samples_per_label):
            lightest_fold = n_samples_per_fold.argmin()
            n_samples_per_fold[lightest_fold] += weight
            label_to_fold[indices[label_index]] = lightest_fold

        fold_labels = label_to_fold[labels]

        for fold_idx in range(self.num_folds):
            test_indices = fold_labels.find(fold_labels == fold_idx)
            train_indices = fold_labels.find(fold_labels != fold_idx)
            self._ts_idx.append(CArray(test_indices))
            self._tr_idx.append(CArray(train_indices))

        return self
