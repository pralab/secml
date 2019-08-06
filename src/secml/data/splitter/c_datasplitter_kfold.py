"""
.. module:: CDataSplitterKFold
   :synopsis: K-Fold splitting

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn.model_selection import KFold

from secml.array import CArray
from secml.data.splitter import CDataSplitter


class CDataSplitterKFold(CDataSplitter):
    """K-Folds dataset splitting.

    Provides train/test indices to split data in train and test sets.
    Split dataset into 'num_folds' consecutive folds (with shuffling).

    Each fold is then used a validation set once while the k - 1
    remaining fold form the training set.

    Parameters
    ----------
    num_folds : int, optional
        Number of folds to create. Default 3.
        This correspond to the size of tr_idx and ts_idx lists.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, is the RandomState instance used by np.random.

    Attributes
    ----------
    class_type : 'kfold'

    Examples
    --------
    >>> from secml.data import CDataset
    >>> from secml.data.splitter import CDataSplitterKFold

    >>> ds = CDataset([[1,2],[3,4],[5,6]],[1,0,1])
    >>> kfold = CDataSplitterKFold(num_folds=3, random_state=0).compute_indices(ds)
    >>> print(kfold.num_folds)
    3
    >>> print(kfold.tr_idx)
    [CArray(2,)(dense: [0 1]), CArray(2,)(dense: [0 2]), CArray(2,)(dense: [1 2])]
    >>> print(kfold.ts_idx)
    [CArray(1,)(dense: [2]), CArray(1,)(dense: [1]), CArray(1,)(dense: [0])]

    """
    __class_type = 'kfold'

    def __init__(self, num_folds=3, random_state=None):

        super(CDataSplitterKFold, self).__init__(
            num_folds=num_folds, random_state=random_state)

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

        sk_splitter = KFold(n_splits=self.num_folds,
                            shuffle=True,
                            random_state=self.random_state)

        # We take sklearn indices (iterators) and map to list of CArrays
        for train_index, test_index in \
                sk_splitter.split(dataset.X.get_data()):
            train_index = CArray(train_index)
            test_index = CArray(test_index)
            self._tr_idx.append(train_index)
            self._ts_idx.append(test_index)

        return self
