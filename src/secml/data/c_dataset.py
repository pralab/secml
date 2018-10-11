"""
.. module:: Dataset
   :synopsis: A dataset with an array of patterns and corresponding labels

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Davide Maiorca <davide.maiorca@diee.unica.it>

"""
from secml.array import CArray
from secml.core.attr_utils import extract_attr


class CDataset(object):
    """Creates a new dataset.

        A dataset consists in a 2-Dimensional patterns array,
        dense or sparse format, with one pattern for each row
        and a flat dense array with each pattern's label.

        Parameters
        ----------
        X : array_like
            Dataset patterns, one for each row. Can be any array-like
            object or a CArray of dense or sparse format. Data is
            converted to 2-Dimensions before storing.
        Y : array_like, optional
            Dataset labels. Can be any 1-Dimensional array-like
            object or any CArray. Data is converted to dense format
            and flattened before storing.
        kwargs : any, optional
            Any other attribute of the dataset.

        Returns
        -------
        out_ds : CDataset
            Dataset consting in a 2-Dimensional patterns array and
            a dense flat vector of corresponding labels (if provided).

        Notes
        -----
        Asymmetric data storing is available, meaning that we do not
        control if the number of stored patters is equal to the
        number of stored labels. However, quite a few functions will
        not work properly if dataset.num_patterns != dataset.num_labels.

        Examples
        --------
        >>> from secml.data import CDataset

        >>> ds = CDataset([[1,2],[3,4],[5,6]],[1,0,1])
        >>> print ds.X
        CArray([[1 2]
         [3 4]
         [5 6]])
        >>> print ds.Y
        CArray([1 0 1])

        >>> ds = CDataset([1,2,3],1)  # Patterns will be converted to 2-Dims
        >>> print ds.X
        CArray([[1 2 3]])
        >>> print ds.Y
        CArray([1])

        >>> from secml.array import CArray
        >>> ds = CDataset(CArray([[1,0],[0,4],[1,0]],tosparse=True), CArray([1,0,1], tosparse=True))
        >>> print ds.X  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (1, 1)	4
          (2, 0)	1)
        >>> print ds.Y
        CArray([1 0 1])

        >>> ds = CDataset([1,2,3],1, id='mydataset', age=34)  # 2 custom attributes
        >>> print ds.id
        mydataset
        >>> print ds.age
        34

        """

    def __init__(self, X, Y=None, **kwargs):
        # Data is forced to be 2-Dimensional (one row for each pattern)
        self._X = CArray(X).atleast_2d()
        # This double casting is to prevent storing a single scalar,
        # should have minimal effect on performance
        self._Y = None if Y is None else CArray(CArray(Y).todense().ravel())
        # Setting any other dataset attribute
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @property
    def num_samples(self):
        """Returns dataset's number of patterns."""
        return self.X.shape[0]

    @property
    def num_features(self):
        """Returns dataset's patterns number of features.

        Number of features should be equal for each pattern.

        """
        return self.X.shape[1]

    @property
    def num_labels(self):
        """Returns dataset's number of labels.

        This can be actually different from dataset.num_patterns.

        """
        return 0 if self.Y is None else self.Y.size

    @property
    def classes(self):
        """Returns dataset's classes (unique).

        Each different labels vector element defines a class.

        """
        return None if self.Y is None else self.Y.unique()

    @property
    def num_classes(self):
        """Returns dataset's number of classes.

        Each different labels vector element defines a class.

        """
        return 0 if self.Y is None else self.classes.size

    @property
    def X(self):
        """Dataset Patterns."""
        return self._X

    @X.setter
    def X(self, value):
        """Set Dataset Patterns.

        Parameters
        ----------
        value : array_like
            Any array-like object or a CArray of dense or
            sparse format. Data is converted to 2-Dimensions
            before storing.

        """
        self._X = CArray(value).atleast_2d()

    @property
    def Y(self):
        """Dataset Labels."""
        return self._Y

    @Y.setter
    def Y(self, value):
        """Set Dataset Labels.

        Parameters
        ----------
        value : array_like
            Any 1-Dimensional array-like object or any CArray.
            Data is converted to dense format and flattened
            before storing.

        """
        # This double casting is to prevent storing a single scalar,
        # should have minimal effect on performance
        self._Y = CArray(CArray(value).todense().ravel())

    @property
    def issparse(self):
        """Return True if dataset's patterns are stored in sparse format, else False."""
        return self.X.issparse

    @property
    def isdense(self):
        """Return True if dataset's patterns are stored in dense format, else False."""
        return self.X.isdense

    def get_params(self):
        """Returns dataset's custom attributes dictionary."""
        # We extract the PUBLIC (pub) attributes from the class dictionary
        return dict((k, getattr(self, k)) for k in extract_attr(self, 'pub'))

    # TODO: ADD DOCSTRING, EXAMPLES
    def __getitem__(self, idx):
        """Given an index, get the corresponding X and Y elements."""
        if not isinstance(idx, tuple) or len(idx) != self.X.ndim:
            raise IndexError("{:} sequences are required for dataset indexing.".format(self.X.ndim))
        ds = self.__class__(self.X.__getitem__(idx), **self.get_params())
        # We now extract the labels corresponding to extracted patterns
        if self.Y is not None:
            ds.Y = self.Y.__getitem__([idx[0] if isinstance(idx, tuple) else idx][0])
        return ds

    # TODO: ADD DOCSTRING, EXAMPLES
    def __setitem__(self, idx, data):
        """Given an index, set the corresponding X and Y elements."""
        if not isinstance(data, self.__class__):
            raise TypeError("dataset can be set only using another dataset.")
        if not isinstance(idx, tuple) or len(idx) != self.X.ndim:
            raise IndexError("{:} sequences are required for dataset indexing.".format(self.X.ndim))
        self.X.__setitem__(idx, data.X)
        # We now set the labels corresponding to set patterns
        if self.Y is None:
            self.Y = data.Y
        elif data.Y is not None:
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
            be converted to sparse and viceversa.

        Returns
        -------
        out_append : CDataset
            A new Dataset resulting of appending new data to
            existing data. Format of resulting dataset is equal
            to current dataset format.

        Notes
        -----
        Append does not occur in-place: a new dataset is allocated
        and filled.

        See Also
        --------
        CArray.append : More informations about arrays append.

        Examples
        --------
        >>> from secml.data import CDataset

        >>> ds = CDataset([[1,2],[3,4],[5,6]],[1,0,1])
        >>> ds_new = ds.append(CDataset([[10,20],[30,40],[50,60]],[1,0,1]))
        >>> print ds_new.X
        CArray([[ 1  2]
         [ 3  4]
         [ 5  6]
         [10 20]
         [30 40]
         [50 60]])
        >>> print ds_new.Y
        CArray([1 0 1 1 0 1])

        >>> ds_new = ds.append(CDataset([[10,20],[30,40],[50,60]],[1,0,1]).tosparse())
        >>> print ds_new.X
        CArray([[ 1  2]
         [ 3  4]
         [ 5  6]
         [10 20]
         [30 40]
         [50 60]])
        >>> print ds_new.Y
        CArray([1 0 1 1 0 1])

        """
        # Format conversion and error checking is managed by CArray.append()
        if dataset.Y is None:
            new_labels = self.Y.deepcopy()
        else:
            new_labels = self.Y.append(dataset.Y)
        return self.__class__(self.X.append(dataset.X, axis=0), new_labels)

    def deepcopy(self):
        """Create a deepcopy of the current dataset.

        Examples
        --------
        >>> from secml.data import CDataset

        >>> ds = CDataset([[1,2],[3,4],[5,6]],[1,0,1])
        >>> ds_copy = ds.deepcopy()
        >>> ds_copy[0, :] = CDataset([[10,20]], 0)

        >>> print ds.X
        CArray([[1 2]
         [3 4]
         [5 6]])
        >>> print ds.Y
        CArray([1 0 1])

        >>> print ds_copy.X
        CArray([[10 20]
         [ 3  4]
         [ 5  6]])
        >>> print ds_copy.Y
        CArray([0 0 1])

        """
        return CDataset(self.X.deepcopy(), self.Y.deepcopy())

    def tosparse(self):
        """Convert dataset's patterns to sparse format.

        Returns
        -------
        out_sparse : CDataset
            A new CDataset with same patterns converted
            to sparse format. Copy is avoided if possible.

        Examples
        --------
        >>> from secml.data import CDataset

        >>> ds = CDataset([[1,2],[3,4],[5,6]],[1,0,1]).tosparse()
        >>> print ds.X  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (0, 1)	2
          (1, 0)	3
          (1, 1)	4
          (2, 0)	5
          (2, 1)	6)
        >>> print ds.Y
        CArray([1 0 1])

        """
        return self.__class__(self.X.tosparse(), self.Y)

    def todense(self):
        """Convert dataset's patterns to dense format.

        Returns
        -------
        out_dense : CDataset
            A new CDataset with same patterns converted
            to dense format. Copy is avoided if possible.

        Examples
        --------
        >>> from secml.data import CDataset

        >>> ds = CDataset(CArray([[1,2],[3,4],[5,6]], tosparse=True),[1,0,1]).todense()
        >>> print ds.X
        CArray([[1 2]
         [3 4]
         [5 6]])

        """
        return self.__class__(self.X.todense(), self.Y)

    def get_labels_asbinary(self, pos_class=None):
        """Return binarized dataset labels.

        Binarizes dataset's labels in [0, 1] range.

        Parameters
        ----------
        pos_class : scalar or str, optional
            Label of the class to consider as positive (label 1).

        Returns
        -------
        binary_labels : CArray
            If pos_class has been defined, flat array with 1 when the
            class label is equal to input positive class's label, else 0.
            If pos_class is None, returns a (num_samples, num_classes) array
            with the dataset labels extended using a one-vs-all scheme.
            If the dataset as no labels set, return None.

        Examples
        --------
        >>> ds = CDataset([[11,22],[33,44],[55,66],[77,88]], [1,0,2,1])
        >>> print ds.get_labels_asbinary(2)
        CArray([0 0 1 0])
        >>> print ds.get_labels_asbinary(1)
        CArray([1 0 0 1])
        >>> print ds.get_labels_asbinary()
        CArray([[0 1 0]
         [1 0 0]
         [0 0 1]
         [0 1 0]])

        >>> ds = CDataset([[11,22],[44,55],[77,88]])
        >>> none_labels = ds.get_labels_asbinary(2)
        >>> none_labels is None
        True

        """
        if pos_class is not None:
            # Assigning 1 for each label of positive class and 0 to all others
            return CArray([1 if e == pos_class else 0 for e in self.Y]) if self.Y is not None else None
        else:  # Return a (num_samples, num_classes) array with OVA labels
            new_labels = CArray.zeros((self.num_samples, self.num_classes),
                                      dtype=self.Y.dtype)
            # We use list of list indexing (find-like)
            new_labels[[list(xrange(self.num_samples)),
                        CArray(self.Y).tolist()]] = 1
            return new_labels

    def get_bounds(self, offset=0.0):
        """Return dataset boundaries plus an offset.

        Parameters
        ----------
        offset : float
            Quantity to be added as an offset. Default 0.

        Returns
        -------
        boundary : List of tuple
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
        for f_idx in xrange(self.num_features):
            boundary.append((x_min[0, f_idx].item(), x_max[0, f_idx].item()))
        return boundary

