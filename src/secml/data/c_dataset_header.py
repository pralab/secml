"""
.. module:: CDatasetHeader
   :synopsis: Header with extra dataset attributes.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.core import CCreator
from secml.core.attr_utils import add_readonly, as_public, extract_attr
from secml.core.type_utils import is_list
from secml.array import CArray
from secml.utils import SubLevelsDict


class CDatasetHeader(CCreator):
    """Creates a new dataset header.

    Each extra keyword argument will be set as a read only attribute.
    To extract a dictionary with the entire set of attributes,
     use `.get_params()`.

    Parameters
    ----------
    kwargs : any, optional
        Any extra attribute of the dataset.

    Examples
    --------
    >>> from secml.data import CDatasetHeader
    >>> from secml.array import CArray

    >>> ds_header = CDatasetHeader(id='mydataset', age=CArray([1,2,3]))
    >>> print(ds_header.id)
    mydataset
    >>> print(ds_header.age)
    CArray([1 2 3])

    """
    __super__ = 'CDatasetHeader'
    __class_type = 'standard'

    def __init__(self, **kwargs):

        self._num_samples = None  # Will be populated by `._validate_params()`
        # Do not create a property for this as will be included in get_params()

        # Set each optional arg as a protected attr and create getter
        for key in kwargs:
            self.add_attr(key, kwargs[key])

    def _validate_params(self):
        """Validate input attributes.

        The following checks will be performed:
         - no attribute should be a list (should be stored as CArray)
         - all CArray must be vector-like and have the same size

        """
        for attr_k, attr_v in self.get_params().items():
            if is_list(attr_v):
                raise TypeError("`list `should be used as a header parameter. "
                                "Use `CArray` instead.")
            if isinstance(attr_v, CArray):
                if not attr_v.is_vector_like:
                    raise ValueError(
                        "`CArray`s should be passed as vector-like.")
                if self._num_samples is not None:
                    if attr_v.size != self._num_samples:
                        raise ValueError(
                            "`{:}` is an array of size {:}. "
                            "{:} expected.".format(attr_k, attr_v.size,
                                                   self._num_samples))
                # Populate the protected _num_samples attribute
                self._num_samples = attr_v.size

    def get_params(self):
        """Returns dataset's custom attributes dictionary."""
        # We extract PUBLIC (pub) + READ/WRITE (rw) + READ ONLY (r)
        return SubLevelsDict((as_public(k), getattr(self, as_public(k)))
                             for k in sorted(extract_attr(self, 'pub+rw+r')))

    def add_attr(self, key, value=None):
        """Add a new attribute to the header.

        Each input attribute will be set as read only (getter only).

        Parameters
        ----------
        key : str
            Attribute to set.
        value : any, optional
            Value to assign to the attribute. If not given, None is used.

        """
        if hasattr(self, key):
            raise AttributeError("attribute '{:}' already defined".format(key))

        # We store lists as CArrays to facilitate indexing
        value = CArray(value) if is_list(value) else value

        # Make sure we store arrays as vector-like
        value = value.ravel() if isinstance(value, CArray) else value

        add_readonly(self, key, value)

        # Make sure that input attributes are consistent
        self._validate_params()

    def __getitem__(self, idx):
        """Given an index, extract the header subset.

        The resulting header will have arrays indexed using input `idx`,
        while other types (dict, scalar, str, etc.) will be passed as is.

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

    def __setstate__(self, state):
        """Reset CDatasetHeader instance after unpickling."""
        self.__dict__.update(state)
        # We now need to reinitialize getters
        for key in state:
            add_readonly(self, as_public(key))

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
            Other attributes are set if not already defined in
            the current header. Otherwise, the value of the attributes in the
            input header should be equal to the value of the same attribute
            in the current header.

        Returns
        -------
        CDatasetHeader

        Notes
        -----
        Append does not occur in-place: a new header is allocated
        and filled.

        See Also
        --------
        CArray.append : More informations about arrays append.

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
