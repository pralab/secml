"""
.. module:: CDataSplitter
   :synopsis: Common interface for dataset splitting

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta, abstractmethod

from secml.core import CCreator


class CDataSplitter(CCreator, metaclass=ABCMeta):
    """Abstract class that defines basic methods for dataset splitting.

    Parameters
    ----------
    num_folds : int, optional
        Number of folds to create. Default 3.
        This corresponds to the size of tr_idx and ts_idx lists.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, is the RandomState instance used by np.random.

    """
    __super__ = 'CDataSplitter'

    def __init__(self, num_folds=3, random_state=None):

        self.num_folds = num_folds
        self.random_state = random_state

        self._tr_idx = []  # Training set indices for each fold
        self._ts_idx = []  # Test set indices for each fold

    @property
    def tr_idx(self):
        """List of training idx obtained with the split of the data."""
        return self._tr_idx

    @property
    def ts_idx(self):
        """List of test idx obtained with the split of the data."""
        return self._ts_idx

    def __iter__(self):
        """Return a train/test indices pair for each fold."""
        for f in range(self.num_folds):
            yield self._tr_idx[f], self._ts_idx[f]

    @abstractmethod
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
        raise NotImplementedError("Each data splitting algorithm must define "
                                  "a `compute_indices` method.")

    def split(self, dataset):
        """Returns a list of split datasets.

        Parameters
        ----------
        dataset : CDataset
            Dataset to split.

        Returns
        -------
        split_ds : list of tuple
            List of tuples (training set, test set), one for each fold.

        """
        # Computing splitting indices
        self.compute_indices(dataset)

        # For each fold, return a tuple (training set, test set)
        ds_list = []
        for tr_idx, ts_idx in self:
            ds_list.append((dataset[tr_idx, :], dataset[ts_idx, :]))

        return ds_list
