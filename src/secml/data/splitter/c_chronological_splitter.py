"""
.. module:: CChronologicalSplitter
   :synopsis: Dataset splitter based on timestamps

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from dateutil import parser
from datetime import datetime

from secml.core import CCreator
from secml.core.type_utils import is_int, is_float
from secml.array import CArray
from secml.data import CDataset


class CChronologicalSplitter(CCreator):
    """Dataset splitter based on timestamps.

    Split dataset into train and test subsets,
     using a timestamp as split point.

    A dataset containing `timestamp` and `timestamp_fmt`
    header attributes is required.

    Parameters
    ----------
    th_timestamp : str
        The split point in time between training and test set.
        Samples having `timestamp <= th_timestamp` will be put in the
        training set, while samples with `timestamp > th_timestamp`
        will be used for the test set. The timestamp must follow the
        ISO 8601 format. Any incomplete timestamp will be parsed too.
    train_size : float or int, optional
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the samples having `timestamp <= th_timestamp`
        to include in the train split. Default 1.0.
        If int, represents the absolute number of train samples.
    test_size : float or int, optional
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the samples having `timestamp > th_timestamp`
        to include in the test split. Default 1.0.
        If int, represents the absolute number of test samples.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, is the RandomState instance used by np.random.
    shuffle : bool, optional
        Whether or not to shuffle the data before splitting.
        If shuffle=False then stratify must be None. Default True.

    """

    def __init__(self, th_timestamp, train_size=1.0, test_size=1.0,
                 random_state=None, shuffle=True):

        if (is_float(test_size) and (test_size <= 0 or test_size > 1.0)) or \
                (is_float(train_size) and (train_size <= 0 or train_size > 1.0)):
            raise ValueError("`test_size` and `train_size` "
                             "must be between (0 and 1.0] if float")

        # We use dateutil.parser is order to allow incomplete
        # timestamps (e.g. a single year '2016')
        self.th_timestamp = parser.isoparse(th_timestamp)

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
        """Compute training set and test set indices.

        Parameters
        ----------
        dataset : CDataset
            Dataset to split.

        Returns
        -------
        tr_idx, ts_idx : CArray
            Flat arrays with the tr/ts indices.

        """
        if not hasattr(dataset.header, 'timestamp') or \
                not hasattr(dataset.header, 'timestamp_fmt'):
            raise AttributeError("dataset must contain `timestamp` and "
                                 "'timestamp_fmt' information")

        timestamps = dataset.header.timestamp
        fmt = dataset.header.timestamp_fmt

        # Pick the samples having `timestamp <= th` to build the training set
        tr_mask = CArray(list(map(
            lambda tstmp: datetime.strptime(tstmp, fmt) <= self.th_timestamp,
            timestamps)))
        # Test set samples are all the other samples
        ts_mask = tr_mask.logical_not()

        # Compute the number of train/test samples
        max_tr = tr_mask.sum()
        max_ts = dataset.num_samples - max_tr

        if max_tr == 0:
            raise ValueError("no samples with timestamp <= {:}. "
                             "Cannot split dataset.".format(self.th_timestamp))

        if max_ts == 0:
            raise ValueError("no samples with timestamp > {:}. "
                             "Cannot split dataset.".format(self.th_timestamp))

        # Compute the actual number of desired train/test samples

        if is_int(self.train_size):
            if self.train_size < 1 or self.train_size > max_tr:
                raise ValueError(
                    "train_size should be between 1 and {:}".format(max_tr))
            else:  # train_size is a valid integer, use it directly
                tr_size = self.train_size
        else:  # Compute the proportion of train samples (at least 1)
            tr_size = int(max(1, round(max_tr * self.train_size)))

        if is_int(self.test_size):
            if self.test_size < 1 or self.test_size > max_ts:
                raise ValueError(
                    "test_size should be between 1 and {:}".format(max_ts))
            else:  # test_size is a valid integer, use it directly
                ts_size = self.test_size
        else:  # Compute the proportion of train samples (at least 1)
            ts_size = int(max(1, round(max_ts * self.test_size)))

        # Get the indices of samples from boolean masks
        tr_idx = CArray(tr_mask.find(tr_mask))
        ts_idx = CArray(ts_mask.find(ts_mask))

        # Get the subset of indices to include in train/test set
        # If shuffle is True, randomize the indices

        if self.shuffle is True:
            tr_idx = CArray.randsample(
                tr_idx, shape=(tr_size, ), random_state=self.random_state)
            ts_idx = CArray.randsample(
                ts_idx, shape=(ts_size, ), random_state=self.random_state)
        else:  # Just slice the arrays of indices
            tr_idx = tr_idx[:tr_size]
            ts_idx = ts_idx[:ts_size]

        self._tr_idx = tr_idx
        self._ts_idx = ts_idx

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
        if not hasattr(dataset.header, 'timestamp') or \
                not hasattr(dataset.header, 'timestamp_fmt'):
            raise AttributeError("dataset must contain `timestamp` and "
                                 "'timestamp_fmt' information")

        # Computing splitting indices
        tr_idx, ts_idx = self.compute_indices(dataset)

        return dataset[tr_idx, :], dataset[ts_idx, :]
