"""
.. module:: CTrainTestSplit
   :synopsis: Train and Test Sets splitter permutation splitting

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn.model_selection import train_test_split

from secml.core import CCreator
from secml.core.type_utils import is_int, is_float
from secml.array import CArray
from secml.data import CDataset


class CTrainTestSplit(CCreator):
    """Train and Test Sets splitter.

    Split dataset into random train and test subsets.

    Quick utility that wraps CDataSplitterShuffle().compute_indices(ds))
    for splitting (and optionally subsampling) data in a oneliner.

    Parameters
    ----------
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
    shuffle : bool, optional
        Whether or not to shuffle the data before splitting.
        If shuffle=False then stratify must be None. Default True.

    Notes
    -----
    train_size and test_size could not be both None. If one is
    set to None the other should be a float, representing a
    percentage, or an integer.

    Examples
    --------
    >>> from secml.data import CDataset
    >>> from secml.data.splitter import CTrainTestSplit

    >>> ds = CDataset([[1,2],[3,4],[5,6],[7,8]],[1,0,1,1])
    >>> tr, ts = CTrainTestSplit(train_size=0.5, random_state=0).split(ds)
    >>> tr.num_samples
    2
    >>> ts.num_samples
    2

    >>> # Get splitting indices without shuffle
    >>> tr_idx, ts_idx = CTrainTestSplit(train_size=0.25,
    ...     random_state=0, shuffle=False).compute_indices(ds)
    >>> tr_idx
    CArray(1,)(dense: [0])
    >>> ts_idx
    CArray(3,)(dense: [1 2 3])

    >>> # At least one sample is needed for each set
    >>> tr, ts = CTrainTestSplit(train_size=0.2, random_state=0).split(ds)
    Traceback (most recent call last):
        ...
    ValueError: train_size should be at least 1 or 0.25

    """

    def __init__(self, train_size=None, test_size=None,
                 random_state=None, shuffle=True):

        if train_size is None and test_size is None:
            raise ValueError(
                "'train_size' and 'test_size' cannot be both None")

        self.train_size = train_size
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle

        self._tr_idx = None  # Training set indices
        self._ts_idx = None  # Test set indices

    @property
    def tr_idx(self):
        """Training set indices obtained with the split of the data."""
        return self._tr_idx

    @property
    def ts_idx(self):
        """Test set indices obtained with the split of the data."""
        return self._ts_idx

    def compute_indices(self, dataset):
        """Compute training set and test set indices for each fold.

        Parameters
        ----------
        dataset : CDataset
            Dataset to split.

        Returns
        -------
        tr_idx, ts_idx : CArray
            Flat arrays with the tr/ts indices.

        """
        min_set_perc = 1 / dataset.num_samples
        if (is_float(self.train_size) and self.train_size < min_set_perc) or \
                (is_int(self.train_size) and self.train_size < 1):
            raise ValueError(
                "train_size should be at least 1 or {:}".format(min_set_perc))
        if (is_float(self.test_size) and self.test_size < min_set_perc) or \
                (is_int(self.test_size) and self.test_size < 1):
            raise ValueError(
                "test_size should be at least 1 or {:}".format(min_set_perc))

        tr_idx, ts_idx = train_test_split(
            CArray.arange(dataset.num_samples).tondarray(),
            train_size=self.train_size,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle)

        self._tr_idx = CArray(tr_idx)
        self._ts_idx = CArray(ts_idx)

        return self.tr_idx, self.ts_idx

    def split(self, dataset):
        """Split dataset into training set and test set.

        Parameters
        ----------
        dataset : CDataset
            Dataset to split.

        Returns
        -------
        ds_train, ds_test : CDataset
            Train and Test datasets.

        """
        # Computing splitting indices
        tr_idx, ts_idx = self.compute_indices(dataset)

        return dataset[tr_idx, :], dataset[ts_idx, :]
