"""
.. module:: CDataSplitterOpenWorldKFold
   :synopsis: Open World K-Fold splitting

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray
from secml.data.splitter import CDataSplitter


class CDataSplitterOpenWorldKFold(CDataSplitter):
    """Open World K-Folds dataset splitting.

    Provides train/test indices to split data in train and test sets.

    In an Open World setting, half (or custom number) of the dataset
    classes are used for training, while all dataset classes are tested.

    Split dataset into 'num_folds' consecutive folds (with shuffling).

    Each fold is then used a validation set once while the k - 1
    remaining fold form the training set.

    Parameters
    ----------
    num_folds : int, optional
        Number of folds to create. Default 3.
        This correspond to the size of tr_idx and ts_idx lists.
    n_train_samples : int, optional
        Number of training samples per client. Default 5.
    n_train_classes : int or None
        Number of dataset classes to use as training.
        If not specified half of dataset classes are used (floored).
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, is the RandomState instance used by np.random.

    Attributes
    ----------
    class_type : 'open-world-kfold'

    Examples
    --------
    >>> from secml.data import CDataset
    >>> from secml.data.splitter import CDataSplitterOpenWorldKFold

    >>> ds = CDataset([[1,2],[3,4],[5,6],[10,20],[30,40],[50,60],
    ...                [100,200],[300,400]],[1,0,1,2,0,1,0,2])
    >>> kfold = CDataSplitterOpenWorldKFold(
    ...     num_folds=3, n_train_samples=2, random_state=0).compute_indices(ds)
    >>> kfold.num_folds
    3
    >>> print(kfold.tr_idx)
    [CArray(2,)(dense: [2 5]), CArray(2,)(dense: [1 4]), CArray(2,)(dense: [0 2])]
    >>> print(kfold.ts_idx)
    [CArray(6,)(dense: [0 1 3 4 6 7]), CArray(6,)(dense: [0 2 3 5 6 7]), CArray(6,)(dense: [1 3 4 5 6 7])]
    >>> print(kfold.tr_classes)  # Class 2 is skipped as there are not enough samples (at least 3)
    [CArray(1,)(dense: [1]), CArray(1,)(dense: [0]), CArray(1,)(dense: [1])]

    """
    __class_type = 'open-world-kfold'

    def __init__(self, num_folds=3, n_train_samples=5,
                 n_train_classes=None, random_state=None):

        super(CDataSplitterOpenWorldKFold, self).__init__(
            num_folds=num_folds, random_state=random_state)

        self.n_train_samples = n_train_samples
        self.n_train_classes = n_train_classes

        self._tr_classes = []

    @property
    def tr_classes(self):
        """List of training classes obtained with the split of the data."""
        return self._tr_classes

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
        self._tr_classes = []

        # If no custom number of training classes is selected,
        # use half of the classes
        n_train_classes = int(dataset.num_classes / 2) \
            if self.n_train_classes is None else int(self.n_train_classes)

        for fold in range(self.num_folds):

            if self.random_state is not None:
                # Adding 1234 to specified random state to get different folds
                random_state = self.random_state + 1234 * fold
            else:  # Random state is None, numpy will manage it
                random_state = self.random_state

            # only 'n_train_classes' random classes will be trained...
            # but now we randsample all classes to backup in case one or
            # more classes will be skipped for n_train_samples
            all_tr_classes = CArray.randsample(dataset.classes,
                                               dataset.num_classes,
                                               random_state=random_state)

            # Placeholder for indices of chosen training classes' samples
            train_samples_idx = CArray([], dtype=int)
            train_classes = CArray([], dtype=all_tr_classes.dtype)

            for train_class in all_tr_classes:
                if train_classes.size >= n_train_classes:
                    break  # we reached the desired number of training classes
                # Vector with indices of current client's samples
                client_samples_idx = CArray(
                    dataset.Y.find(dataset.Y == train_class))
                # Check if we have at least n_train_samples + 1 samples for
                # current client
                if client_samples_idx.size < self.n_train_samples + 1:
                    self.logger.warning("skipping class {:} for training set. "
                                        "{:} samples is less than {:}."
                                        "".format(train_class,
                                                  client_samples_idx.size,
                                                  self.n_train_samples + 1))
                    continue

                # Random subselection of training samples
                random_samples = CArray.randsample(client_samples_idx,
                                                   self.n_train_samples,
                                                   random_state=random_state)
                # Appending to vector of indices for training set a random
                # subselection of samples
                train_samples_idx = train_samples_idx.append(random_samples)
                # Adding class id
                train_classes = train_classes.append(train_class)

            # We store the sorted training classes list
            self._tr_classes += [train_classes.sort()]

            # Storing a sorted array of training samples indices
            train_samples_idx.sort(inplace=True)

            # All other samples go to test
            test_samples_idx = CArray(
                [idx for idx in range(dataset.num_samples)
                 if idx not in train_samples_idx])

            self._tr_idx += [train_samples_idx]
            self._ts_idx += [test_samples_idx]

        return self
