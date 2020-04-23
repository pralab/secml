"""
.. module:: CDatasetHeader
   :synopsis: Header with extra dataset attributes.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.core import CCreator
from secml.core.attr_utils import is_writable
from secml.core.type_utils import is_list
from secml.array import CArray


class CDatasetHeader(CCreator):
    """Creates a new dataset header.

    Parameters to be included into the header could be defined as keyword
    init arguments or by setting them as new public header attributes.

    Immutable objects (scalar, string, tuple, dictionary) will be passed
    as they are while indexing the header. Arrays will be indexed and the
    result of indexing will be returned.

    To extract a dictionary with the entire set of attributes,
     use `.get_params()`.

    Parameters
    ----------
    kwargs : any, optional
        Any extra attribute of the dataset.
        Could be an immutable object (scalar, tuple, dict, str),
        or a vector-like CArray. Lists are automatically converted
        to vector-like CArrays.

    Examples
    --------
    >>> from secml.data import CDatasetHeader
    >>> from secml.array import CArray

    >>> ds_header = CDatasetHeader(id='mydataset', colors=CArray([1,2,3]))

    >>> print(ds_header.id)
    mydataset
    >>> print(ds_header.colors)
    CArray([1 2 3])

    >>> ds_header.age = 32
    >>> print(ds_header.age)
    32

    """
    __super__ = 'CDatasetHeader'
    __class_type = 'standard'

    def __init__(self, **kwargs):

        self._num_samples = None  # Will be populated by `._validate_params()`

        # Set each optional arg
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def num_samples(self):
        """The number of samples for which the header defines extra params."""
        return self._num_samples

    def __setattr__(self, key, value):
        """Add a new attribute to the header.

        Parameters
        ----------
        key : str
            Attribute to set.
        value : any
            Value to assign to the attribute.
            Could be an immutable object (scalar, tuple, dict, str),
            or a vector-like CArray. Lists are automatically converted
            to vector-like CArrays.

        """
        # We store lists as CArrays to facilitate indexing
        value = CArray(value) if is_list(value) else value

        # Make sure we store arrays as vector-like
        value = value.ravel() if isinstance(value, CArray) else value

        super(CDatasetHeader, self).__setattr__(key, value)

        # Make sure that input writable attributes are consistent
        if is_writable(self, key):
            self._validate_params()

    def _validate_params(self):
        """Validate input attributes.

        The following checks will be performed:
         - all CArray must have the same size

        """
        for attr_k, attr_v in self.get_params().items():
            if isinstance(attr_v, CArray):
                if self.num_samples is not None:
                    if attr_v.size != self.num_samples:
                        delattr(self, attr_k)  # Remove faulty attribute
                        raise ValueError(
                            "`{:}` is an array of size {:}. "
                            "{:} expected.".format(attr_k, attr_v.size,
                                                   self.num_samples))
                # Populate the protected _num_samples attribute
                self._num_samples = attr_v.size

    def __getitem__(self, idx):
        """Given an index, extract the header subset.

        Immutable objects (scalar, string, tuple, dictionary) will be passed
        as they are while indexing the header. Arrays will be indexed and the
        result of indexing will be returned.

        Examples
        --------
        >>> from secml.data import CDatasetHeader
        >>> from secml.array import CArray

        >>> ds_header = CDatasetHeader(id='mydataset', age=CArray([1,2,3]))

        >>> h_subset = ds_header[[0, 2]]
        >>> h_subset.id
        'mydataset'
        >>> h_subset.age
        CArray(2,)(dense: [1 3])

        """
        subset = dict()
        for attr in self.get_params():
            if isinstance(getattr(self, attr), CArray):
                subset[attr] = getattr(self, attr)[idx]
            else:  # Pass other types (dict, scalar, str, ...) as is
                subset[attr] = getattr(self, attr)

        return self.__class__(**subset)

    def __str__(self):
        if len(self.get_params()) == 0:
            return self.__class__.__name__ + "{}"
        return self.__class__.__name__ + \
            "{'" + "', '".join(self.get_params()) + "'}"

    def append(self, header):
        """Append input header to current header.

        Parameters
        ----------
        header : CDatasetHeader
            Header to append. Only attributes which are arrays are merged.
            Other attributes are set if not already defined in the current
            header. Otherwise, the value of the attributes in the input
            header should be equal to the value of the same attribute
            in the current header.

        Returns
        -------
        CDatasetHeader

        Notes
        -----
        Append does not occur in-place: a new header is allocated and filled.

        See Also
        --------
        CArray.append : More information about arrays append.

        Examples
        --------
        >>> from secml.data import CDatasetHeader
        >>> from secml.array import CArray

        >>> ds_header1 = CDatasetHeader(id={'a': 0, 'b': 2}, a=2, age=CArray([1,2,3]))
        >>> ds_header2 = CDatasetHeader(id={'a': 0, 'b': 2}, b=4, age=CArray([1,2,3]))

        >>> ds_merged = ds_header1.append(ds_header2)
        >>> ds_merged.age
        CArray(6,)(dense: [1 2 3 1 2 3])
        >>> ds_merged.id  # doctest: +SKIP
        {'a': 0, 'b': 2}
        >>> ds_merged.a
        2
        >>> ds_merged.b
        4

        """
        subset = dict()
        for attr in header.get_params():
            if hasattr(self, attr):  # Attribute already in current header
                if isinstance(getattr(self, attr), CArray):
                    subset[attr] = getattr(self, attr)\
                        .append(getattr(header, attr))
                elif getattr(self, attr) != getattr(header, attr):
                    # For not-arrays, we check equality
                    raise ValueError(
                        "value of '{:}' in input header should be equal "
                        "to '{:}'".format(attr, getattr(self, attr)))
            else:  # New attribute in input header
                subset[attr] = getattr(header, attr)

        # Append attributes which are not in the input header
        for attr in self.get_params():
            if attr not in subset:
                subset[attr] = getattr(self, attr)

        return self.__class__(**subset)
