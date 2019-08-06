"""
.. module:: CDataSplitterShuffle
   :synopsis: Random permutation splitting

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn.model_selection import ShuffleSplit

from secml.array import CArray
from secml.data.splitter import CDataSplitter


class CDataSplitterShuffle(CDataSplitter):
    """Random permutation dataset splitting.

    Yields indices to split data into training and test sets.

    Note: contrary to other dataset splitting strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Parameters
    ----------
    num_folds : int, optional
        Number of folds to create. Default 3.
        This correspond to the size of tr_idx and ts_idx lists.
    train_size : float, int, or None, optional
        If None (default), the value is automatically set to the
        complement of the test size. If float, should be between
        0.0 and 1.0 and represent the proportion of the dataset
        to include in the train split. If int, represents the
        absolute number of train samples.
    test_size : float, int, or None, optional
        If None (default), the value is automatically set to the
        complement of the train size. If float, should be between
        0.0 and 1.0 and represent the proportion of the dataset
        to include in the test split. If int, represents the
        absolute number of test samples.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, is the RandomState instance used by np.random.

    Attributes
    ----------
    class_type : 'shuffle'

    Notes
    -----
    train_size and test_size could not be both None. If one is
    set to None the other should be a float, representing a
    percentage, or an integer.

    Examples
    --------
    >>> from secml.data import CDataset
    >>> from secml.data.splitter import CDataSplitterShuffle

    >>> ds = CDataset([[1,2],[3,4],[5,6]],[1,0,1])
    >>> shuffle = CDataSplitterShuffle(num_folds=3, train_size=0.5, random_state=0).compute_indices(ds)
    >>> shuffle.num_folds
    3
    >>> shuffle.tr_idx
    [CArray(1,)(dense: [0]), CArray(1,)(dense: [1]), CArray(1,)(dense: [1])]
    >>> shuffle.ts_idx
    [CArray(2,)(dense: [2 1]), CArray(2,)(dense: [2 0]), CArray(2,)(dense: [0 2])]

    >>> # Setting the train_size or the test_size to an arbitrary percentage
    >>> shuffle = CDataSplitterShuffle(num_folds=3, train_size=0.2, random_state=0).compute_indices(ds)
    >>> shuffle.num_folds
    3
    >>> shuffle.tr_idx
    [CArray(0,)(dense: []), CArray(0,)(dense: []), CArray(0,)(dense: [])]
    >>> shuffle.ts_idx
    [CArray(3,)(dense: [2 1 0]), CArray(3,)(dense: [2 0 1]), CArray(3,)(dense: [0 2 1])]

    """
    __class_type = 'shuffle'

    def __init__(self,  num_folds=3, train_size=None,
                 test_size=None, random_state=None):

        super(CDataSplitterShuffle, self).__init__(
            num_folds=num_folds, random_state=random_state)

        self.train_size = train_size
        self.test_size = test_size

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

        sk_splitter = ShuffleSplit(n_splits=self.num_folds,
                                   train_size=self.train_size,
                                   test_size=self.test_size,
                                   random_state=self.random_state)

        # We take sklearn indices (iterators) and map to list of CArrays
        for train_index, test_index in \
                sk_splitter.split(dataset.X.get_data()):
            train_index = CArray(train_index)
            test_index = CArray(test_index)
            self._tr_idx.append(train_index)
            self._ts_idx.append(test_index)

        return self
