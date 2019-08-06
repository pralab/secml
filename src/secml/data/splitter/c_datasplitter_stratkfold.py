"""
.. module:: CDataSplitterStratifiedKFold
   :synopsis: Stratified K-Fold

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn.model_selection import StratifiedKFold

from secml.array import CArray
from secml.data.splitter import CDataSplitter


class CDataSplitterStratifiedKFold(CDataSplitter):
    """Stratified K-Folds dataset splitting.

    Provides train/test indices to split data in train test sets.

    This dataset splitting object is a variation of KFold, which
    returns stratified folds. The folds are made by preserving
    the percentage of samples for each class.

    Parameters
    ----------
    num_folds : int, optional
        Number of folds to create. Default 3.
        This correspond to the size of tr_idx and ts_idx lists.
        For stratified K-Fold, this cannot be higher than the
        minimum number of samples per class in the dataset.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, is the RandomState instance used by np.random.

    Attributes
    ----------
    class_type : 'strat-kfold'

    Examples
    --------
    >>> from secml.data import CDataset
    >>> from secml.data.splitter import CDataSplitterStratifiedKFold

    >>> ds = CDataset([[1,2],[3,4],[5,6],[7,8]],[1,0,0,1])
    >>> stratkfold = CDataSplitterStratifiedKFold(num_folds=2, random_state=0).compute_indices(ds)
    >>> stratkfold.num_folds  # Cannot be higher than the number of samples per class
    2
    >>> stratkfold.tr_idx
    [CArray(2,)(dense: [1 3]), CArray(2,)(dense: [0 2])]
    >>> stratkfold.ts_idx
    [CArray(2,)(dense: [0 2]), CArray(2,)(dense: [1 3])]

    """
    __class_type = 'strat-kfold'

    def __init__(self, num_folds=3, random_state=None):

        super(CDataSplitterStratifiedKFold, self).__init__(
            num_folds, random_state=random_state)

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

        sk_splitter = StratifiedKFold(n_splits=self.num_folds,
                                      shuffle=True,
                                      random_state=self.random_state)

        # We take sklearn indices (iterators) and map to list of CArrays
        for train_index, test_index in \
                sk_splitter.split(X=dataset.X.get_data(),
                                  y=dataset.Y.get_data()):
            train_index = CArray(train_index)
            test_index = CArray(test_index)
            self._tr_idx.append(train_index)
            self._ts_idx.append(test_index)

        return self
