"""
.. module:: CDataset
   :synopsis: A dataset with an array of patterns and corresponding labels

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.core import CCreator
from secml.array import CArray
from secml.data import CDatasetHeader
from secml.data.data_utils import label_binarize_onehot


class CDataset(CCreator):
    """Creates a new dataset.

    A dataset consists in a 2-Dimensional patterns array,
    dense or sparse format, with one pattern for each row
    and a flat dense array with each pattern's label.

    Parameters
    ----------
    x : `array_like` or CArray
        Dataset patterns, one for each row.
        Array is converted to 2-Dimensions before storing.
    y : `array_like` or CArray
        Dataset labels. Array is converted to dense format
        and flattened before storing.
    header : CDatasetHeader or None, optional
        The header for the dataset. Will define any extra parameter.
        See `CDatasetHeader` docs for more information.

    Examples
    --------
    >>> from secml.data import CDataset

    >>> ds = CDataset([[1,2],[3,4],[5,6]],[1,0,1])
    >>> print(ds.X)
    CArray([[1 2]
     [3 4]
     [5 6]])
    >>> print(ds.Y)
    CArray([1 0 1])

    >>> ds = CDataset([1,2,3],1)  # Patterns will be converted to 2-Dims
    >>> print(ds.X)
    CArray([[1 2 3]])
    >>> print(ds.Y)
    CArray([1])

    >>> from secml.array import CArray
    >>> ds = CDataset(CArray([[1,0],[0,4],[1,0]],tosparse=True), CArray([1,0,1],tosparse=True))
    >>> print(ds.X)  # doctest: +NORMALIZE_WHITESPACE
    CArray(  (0, 0)	1
      (1, 1)	4
      (2, 0)	1)
    >>> print(ds.Y)
    CArray([1 0 1])

    The number of labels must be equal to the number of samples

    >>> ds = CDataset([[1,2],[3,4]],1)
    Traceback (most recent call last):
     ...
    ValueError: number of labels (1) must be equal to the number of samples (2).

    >>> from secml.data import CDatasetHeader
    >>> ds = CDataset([1,2,3], 1, CDatasetHeader(id='mydataset', age=34))  # 2 extra attributes
    >>> print(ds.header.id)
    mydataset
    >>> print(ds.header.age)
    34

    """
    __super__ = 'CDataset'
    __class_type = 'standard'

    def __init__(self, x, y, header=None):

        # Default placeholders (keep to make _check_samples_labels work)
        self._X = None
        self._Y = None

        # Patterns are forced to be 2-D (one row for each pattern)
        self.X = x
        # Labels are 1-D (one label for each sample)
        # This will also check patterns/labels size consistency
        self.Y = y

        # Header that will store extra attributes of dataset
        self.header = header

    def __setstate__(self, state):
        """Reset CDataset instance after unpickling."""
        self.__dict__.update(state)
        # Initialize header placeholder if not available
        # Necessary to unpickle old dataset (stored with secml < v0.6)
        if not hasattr(self, '_header'):
            self._header = None

    @property
    def X(self):
        """Dataset Patterns."""
        return self._X

    @X.setter
    def X(self, value):
        """Set Dataset Patterns.

        Parameters
        ----------
        value : `array_like` or CArray
            Array containing patterns. Data is converted to 2-Dimensions
            before storing.

        """
        x = CArray(value).atleast_2d()
        if self.Y is not None:  # Checking number of samples/labels equality
            self._check_samples_labels(x=x)
        self._X = x

    @property
    def Y(self):
        """Dataset Labels."""
        return self._Y

    @Y.setter
    def Y(self, value):
        """Set Dataset Labels.

        Parameters
        ----------
        value : `array_like` or CArray
            Array containing labels. Array is converted to dense format
            and flattened before storing.

        """
        y = CArray(value).todense().ravel()
        if self._X is not None:  # Checking number of samples/labels equality
            self._check_samples_labels(y=y)
        self._Y = y

    @property
    def header(self):
        """Dataset header."""
        return self._header

    @header.setter
    def header(self, value):
        """Dataset header."""
        if value is not None:
            if not isinstance(value, CDatasetHeader):
                raise TypeError(
                    "'header' must be an instance of 'CDatasetHeader'")

            # Check if header is compatible (same num_samples)
            if value.num_samples is not None and \
                    self.num_samples != value.num_samples:
                raise ValueError(
                    "incompatible header size {:}. {:} expected.".format(
                        self.num_samples, value.num_samples))

        self._header = value

    @property
    def num_samples(self):
        """Number of patterns."""
        return self.X.shape[0]

    @property
    def num_features(self):
        """Number of features."""
        return self.X.shape[1]

    @property
    def num_labels(self):
        """Returns dataset's number of labels."""
        return self.Y.size

    @property
    def classes(self):
        """Classes (unique)."""
        return self.Y.unique()

    @property
    def num_classes(self):
        """Number of classes."""
        return self.classes.size

    @property
    def issparse(self):
        """True if patterns are stored in sparse format, else False."""
        return self.X.issparse

    @property
    def isdense(self):
        """True if patterns are stored in dense format, else False."""
        return self.X.isdense

    def _check_samples_labels(self, x=None, y=None):
        """Raise ValueError if the number of labels is different to
        the number of samples."""
        x = self.X if x is None else x
        y = self.Y if y is None else y
        if x.shape[0] != y.size:
            raise ValueError(
                "number of labels ({:}) must be equal to the number "
                "of samples ({:}).".format(y.size, x.shape[0]))

    def __getitem__(self, idx):
        """Given an index, get the corresponding X and Y elements."""
        if not isinstance(idx, tuple) or len(idx) != self.X.ndim:
            raise IndexError(
                "{:} sequences are required for indexing.".format(self.X.ndim))

        y = self.Y.__getitem__([idx[0] if isinstance(idx, tuple) else idx][0])

        header = None
        if self.header is not None:
            header = self.header.__getitem__(
                [idx[0] if isinstance(idx, tuple) else idx][0])

        return self.__class__(self.X.__getitem__(idx), y, header=header)

    def __setitem__(self, idx, data):
        """Given an index, set the corresponding X and Y elements."""
        if not isinstance(data, self.__class__):
            raise TypeError("dataset can be set only using another dataset.")
        if not isinstance(idx, tuple) or len(idx) != self.X.ndim:
            raise IndexError(
                "{:} sequences are required for indexing.".format(self.X.ndim))
        self.X.__setitem__(idx, data.X)
        # We now set the labels corresponding to set patterns
        self.Y.__setitem__([idx[0] if isinstance(idx, tuple) else idx][0], data.Y)

    def append(self, dataset):
        """Append input dataset to current dataset.

        Parameters
        ----------
        dataset : CDataset
            Dataset to append. Patterns are appended on first
            axis (axis=0) so the number of features must be
            equal to dataset.num_features. If current dataset
            format is sparse, dense dataset to append will
            be converted to sparse and vice-versa.

        Returns
        -------
        CDataset
            A new Dataset resulting of appending new data to
            existing data. Format of resulting dataset is equal
            to current dataset format.

        Notes
        -----
        Append does not occur in-place: a new dataset is allocated
        and filled.

        See Also
        --------
        :meth:`CArray.append` : More information about arrays append.

        Examples
        --------
        >>> from secml.data import CDataset

        >>> ds = CDataset([[1,2],[3,4],[5,6]],[1,0,1])
        >>> ds_new = ds.append(CDataset([[10,20],[30,40],[50,60]],[1,0,1]))
        >>> print(ds_new.X)
        CArray([[ 1  2]
         [ 3  4]
         [ 5  6]
         [10 20]
         [30 40]
         [50 60]])
        >>> print(ds_new.Y)
        CArray([1 0 1 1 0 1])

        >>> ds_new = ds.append(CDataset([[10,20],[30,40],[50,60]],[1,0,1]).tosparse())
        >>> print(ds_new.X)
        CArray([[ 1  2]
         [ 3  4]
         [ 5  6]
         [10 20]
         [30 40]
         [50 60]])
        >>> print(ds_new.Y)
        CArray([1 0 1 1 0 1])

        """
        # Format conversion and error checking is managed by CArray.append()
        new_labels = self.Y.append(dataset.Y)

        # As the input dataset (or self) could have no header, check it
        if dataset.header is None or self.header is None:
            new_header = self.header or dataset.header
            if new_header is not None:
                if new_header.num_samples is not None:
                    raise ValueError(
                        "cannot append a dataset with header and "
                        "{:} samples as the other has no header. "
                        "Define a consistent header for both dataset "
                        "and try again.".format(new_header.num_samples))
        else:  # Both input ds and self have header, merge them
            new_header = self.header.append(dataset.header)

        return self.__class__(
            self.X.append(dataset.X, axis=0), new_labels, header=new_header)

    def tosparse(self):
        """Convert dataset's patterns to sparse format.

        Returns
        -------
        CDataset
            A new CDataset with same patterns converted
            to sparse format. Copy is avoided if possible.

        Examples
        --------
        >>> from secml.data import CDataset

        >>> ds = CDataset([[1,2],[3,4],[5,6]],[1,0,1]).tosparse()
        >>> print(ds.X)  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (0, 1)	2
          (1, 0)	3
          (1, 1)	4
          (2, 0)	5
          (2, 1)	6)
        >>> print(ds.Y)
        CArray([1 0 1])

        """
        return self.__class__(self.X.tosparse(), self.Y, header=self.header)

    def todense(self):
        """Convert dataset's patterns to dense format.

        Returns
        -------
        CDataset
            A new CDataset with same patterns converted
            to dense format. Copy is avoided if possible.

        Examples
        --------
        >>> from secml.data import CDataset

        >>> ds = CDataset(CArray([[1,2],[3,4],[5,6]], tosparse=True),[1,0,1]).todense()
        >>> print(ds.X)
        CArray([[1 2]
         [3 4]
         [5 6]])

        """
        return self.__class__(self.X.todense(), self.Y, header=self.header)

    def get_labels_ovr(self, pos_label):
        """Return dataset labels in one-vs-rest encoding.

        Parameters
        ----------
        pos_label : scalar or str
            Label of the class to consider as positive.

        Returns
        -------
        CArray
            Flat array with 1 when the class label is equal
            to input positive class's label, else 0.

        Examples
        --------
        >>> ds = CDataset([[11,22],[33,44],[55,66],[77,88]], [1,0,2,1])
        >>> print(ds.get_labels_ovr(2))
        CArray([0 0 1 0])
        >>> print(ds.get_labels_ovr(1))
        CArray([1 0 0 1])

        """
        # Assigning 1 for each label of positive class and 0 to all others
        return CArray([1 if e == pos_label else 0 for e in self.Y])

    def get_labels_onehot(self):
        """Return dataset labels in one-hot encoding.

        Returns
        -------
        binary_labels : CArray
            A (num_samples, num_classes) array with the dataset labels
            one-hot encoded.

        Examples
        --------
        >>> ds = CDataset([[11,22],[33,44],[55,66],[77,88]], [1,0,2,1])
        >>> print(ds.get_labels_onehot())
        CArray([[0 1 0]
         [1 0 0]
         [0 0 1]
         [0 1 0]])

        """
        # Our convention is that the labels are from 0 ... N (integers only)
        # Do not use `self.classes` directly as Y can contain only a subset
        # of the known classes
        return label_binarize_onehot(self.Y)

    def get_bounds(self, offset=0.0):
        """Return dataset boundaries plus an offset.

        Parameters
        ----------
        offset : float
            Quantity to be added as an offset. Default 0.

        Returns
        -------
        boundary : list of tuple
            Every tuple contain min and max feature value plus an
            offset for corresponding coordinate.

        Examples
        ----------
        >>> from secml.array import CArray
        >>> from secml.data import  CDataset

        >>> ds = CDataset([[1,2,3],[4,5,6]], [1,2])
        >>> ds.get_bounds()
        [(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)]

        """
        x_min = self.X.min(axis=0) - offset
        x_max = self.X.max(axis=0) + offset
        boundary = []
        for f_idx in range(self.num_features):
            boundary.append((x_min[0, f_idx].item(), x_max[0, f_idx].item()))
        return boundary

