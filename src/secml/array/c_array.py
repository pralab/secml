"""
.. module:: ArrayContainer
   :synopsis: Class that defines and manage data arrays

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from copy import deepcopy
import numpy as np
import scipy.sparse as scs

from c_dense import CDense
from c_sparse import CSparse
from secml.core.type_utils import \
    is_int, is_scalar, is_bool, is_ndarray, is_scsarray, to_builtin

__all__ = ['is_array', 'is_array_buffer']


class CArray(object):
    """Creates an array.

    Data will be stored in dense form by default.

    Parameters
    ----------
    data : array_like or any built-in datatype
        Data to be stored. Can be any array-like structure
        (sparse or dense) or any built-in list, scalar or string.
    dtype : str or dtype, optional
        Typecode or data-type to which the array is cast.
        If None (default), dtype is inferred from input data.
    copy : bool, optional
        If False (default) a reference to input data will be
        stored if possibile. Otherwise, a copy of original data
        is made first. If data is a nested sequence (a list)
        or dtype is different, a copy will be made anyway.
    shape : int or sequence of ints, optional
        Shape of the new array, e.g., '(2, 3)' or '2'.
    tosparse : bool, optional
        If True, input data will be converted to sparse format.
        Otherwise (default), if input is not a CArray, a dense
        array is returned, or if CArray, its format is preserved.

    See Also
    --------
    .CDense : Array container for dense data.
    .CSparse : Array container for sparse data.

    Examples
    --------
    >>> from secml.array import CArray

    >>> print CArray([[1, 2], [3, 4]])
    CArray([[1 2]
     [3 4]])

    >>> print CArray(True)
    CArray([ True])

    >>> print CArray([1,0,3,4], tosparse=True)  # doctest: +NORMALIZE_WHITESPACE
    CArray(  (0, 0)	1
      (0, 2)    3
      (0, 3)    4)

    >>> print CArray([1,2,3], dtype=float, shape=(3,1))  # Custom dtype and shape
    CArray([[ 1.]
     [ 2.]
     [ 3.]])

    """
    __slots__ = '_data'  # CArray has only one slot for the buffer

    def __init__(
            self, data, dtype=None, copy=False, shape=None, tosparse=False):

        # Not implemented operators return NotImplemented
        if data is NotImplemented:
            raise TypeError("operator not implemented")

        # Initialization can be used also for light casting, i.e. cast a CArray
        # back to its same class to assure correct output type. The impact on
        # performance should be minimal...

        if isinstance(data, CArray):
            # Light casting: store data after format conversion
            self._data = data.tosparse()._data if tosparse is True or \
                data.issparse else data.todense()._data
            if copy is True and self.isdense == data.isdense:
                # copy needed and no previous change of format
                self._data = deepcopy(self._data)
            if dtype is not None and self._data.dtype != dtype:
                self._data = self._data.astype(dtype)
        elif tosparse is True or \
                isinstance(data, CSparse) or scs.issparse(data):
            self._data = CSparse(data, dtype, copy, shape)
        else:
            self._data = CDense(data, dtype, copy, shape)

    def get_data(self):
        """Return stored data as a standard dtype.

        Returns
        -------
        out : np.ndarray, scipy.sparse.csr_matrix
            If array is dense, a np.ndarray is returned.
            If array is sparse, a scipy.sparse.csr_matrix is returned.

        See Also
        --------
        `.tondarray()` : returns a np.ndarray, regardless of array format.
        `.tocsr()` : returns a scipy.sparse.csr_matrix, regardless of array format.

        """
        return self.tondarray() if self.isdense is True else self.tocsr()

    def _instance_array(self, data):
        """Returns input data with correct shape.

        For any one-element array, i.e. array of shape (1, ) or (1, 1)
        returns the single object inside it with using a built-in type.

        Parameters
        ----------
        data : array_like, scalar_like, NotImplemented
            Data to be converted. Could be:
             - NotImplemented (raised by not implemented built-in operators)
             - CArray buffers (CDense, CSparse)
             - scalar-like (int, float, str, bool and numpy equivalents)

        Returns
        -------
        out : CArray, scalar_like, NotImplemented
            A scalar-like data-type for any one-element array or scalar.
             Built-in types will be returned (int, bool, float)
            A CArray for any array of size > 1.
            A NotImplemented object (equivalent to False).

        """
        try:
            return to_builtin(data)
        except TypeError:
            # Probably the input is an array buffer or NotImplemented
            pass

        if is_array_buffer(data):  # CDense, CSparse
            out = self.__class__(data)
            if out.size == 1:
                # For (1,) or (1, 1) array return the contained scalar
                return to_builtin(out.tondarray().ravel()[0])
            return out

        elif data is NotImplemented:
            # Returned if a standard operator (+, *, abs, ...) is not supported
            return NotImplemented

        else:  # Unknown object returned by the calling method, raise error
            raise TypeError(
                "objects of type {:} not supported.".format(type(data)))

    # ------------------------------ #
    # # # # # # PROPERTIES # # # # # #
    # -------------------------------#

    @property
    def isdense(self):
        """True if data is stored in DENSE form, False otherwise."""
        return True if isinstance(self._data, CDense) else False

    @property
    def issparse(self):
        """True if data is stored in SPARSE form, False otherwise."""
        return True if isinstance(self._data, CSparse) else False

    @property
    def T(self):
        """Transposed array data.

        See Also
        --------
        .CArray.transpose : bound method to transpose an array.

        """
        return self.__class__(self._data.T)

    @property
    def shape(self):
        """Shape of stored data, tuple of ints."""
        return self._data.shape

    @property
    def size(self):
        """Size (number of elements) of array.

        For sparse data, this counts both zeros and non-zero elements.

        """
        return self._data.size

    @property
    def ndim(self):
        """Number of array dimensions.

        This is always 2 for sparse arrays.

        """
        return self._data.ndim

    @property
    def dtype(self):
        """Data-type of stored data."""
        return self._data.dtype

    @property
    def nnz(self):
        """Number of non-zero array elements.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([1,0,3,0], tosparse=True).nnz
        2

        """
        return self._data.nnz

    @property
    def nnz_indices(self):
        """Index of non-zero array elements.

        Returns
        -------
        nnz_indices : list
            List of 2 lists. Inside out[0] there are
            the indices of the corresponding rows and inside out[1]
            there are the indices of the corresponding columns of
            non-zero array elements.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray([1,0,3,0], tosparse=True)
        >>> nzz_indices = array.nnz_indices
        >>> nzz_indices
        [[0, 0], [0, 2]]
        >>> print array[nzz_indices]  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (0, 1)	3)

        >>> array = CArray([1,0,3,0])
        >>> nzz_indices = array.nnz_indices
        >>> nzz_indices
        [[0, 0], [0, 2]]
        >>> print array[nzz_indices]
        CArray([1 3])

        """
        return self._data.nnz_indices

    @property
    def nnz_data(self):
        """Return non-zero array elements.

        Returns
        -------
        nnz_data : CArray
            Flat array, dense, shape (n, ), with non-zero array elements.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray([1,0,3,0], tosparse=True)
        >>> print array.nnz_data  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (0, 1)	3)

        """
        return self._instance_array(self._data.nnz_data)

    @property
    def is_vector_like(self):
        """True if array is vector-like.

        An array is vector-like when 1-Dimensional or
        2-Dimensional with shape[0] == 1.

        Examples
        --------
        >>> from secml.array import CArray

        >>> a = CArray([1,2,3])
        >>> a.is_vector_like
        True

        >>> a = CArray([1,2,3], tosparse=True)  # sparse arrays always 2-Dimensional
        >>> a.is_vector_like
        True

        >>> a = CArray([[1,2],[3,4]])
        >>> a.is_vector_like
        False

        """
        return True if not (self.ndim > 1 and self.shape[0] > 1) else False

    # --------------------------- #
    # # # # # # CASTING # # # # # #
    # ----------------------------#

    def todense(self, dtype=None, shape=None):
        """Converts array to dense format.

        Return current array if it has already a dense format.

        Parameters
        ----------
        dtype : str or dtype, optional
            Typecode or data-type to which the array is cast.
        shape : sequence of ints, optional
            Shape of the new array, e.g., '(2, 3)'.

        Returns
        -------
        out : CArray
            Dense array with input data and desired dtype and/or shape.

        Notes
        -----
        If current array has already a dense format, `dtype` and `shape`
        parameters will not be functional. Use `.astype()` or `.reshape()`
        function to alter array shape/dtype.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[2, 0], [3, 4]], tosparse=True).todense(dtype=float)
        CArray([[ 2.  0.]
         [ 3.  4.]])

        >>> print CArray([[2, 0], [3, 4]], tosparse=True).todense(shape=(4,))
        CArray([2 0 3 4])

        """
        if self.issparse is False and (shape is not None or dtype is not None):
            raise ValueError("array is already dense. Use astype() or "
                             "reshape() function to alter array shape/dtype.")
        elif self.issparse is True:
            return self.__class__(
                self._data.todense(), shape=shape, dtype=dtype)
        else:
            return self

    def tosparse(self, dtype=None, shape=None):
        """Converts array to sparse format.

        Return current array if it has already a sparse format.

        Parameters
        ----------
        dtype : str or dtype, optional
            Typecode or data-type to which the array is cast.
        shape : sequence of ints, optional
            Shape of the new array, e.g., '(2, 3)'. Only 2-Dimensional
            sparse arrays are supported.

        Returns
        -------
        out : CArray
            Sparse array with input data and desired dtype and/or shape.

        Notes
        -----
        If current array has already a sparse format, `dtype` and `shape`
        parameters will not be functional. Use `.astype()` or `.reshape()`
        function to alter array shape/dtype.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[2, 0], [3, 4]]).tosparse(dtype=float)  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	2.0
          (1, 0)	3.0
          (1, 1)	4.0)

        >>> print CArray([[2, 0], [3, 4]]).tosparse(shape=(1, 4))  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	2
          (0, 2)	3
          (0, 3)	4)

        """
        if self.isdense is False and (shape is not None or dtype is not None):
            raise ValueError("array is already sparse. Use astype() or "
                             "reshape() unction to alter array shape/dtype.")
        elif self.isdense is True:
            return self.__class__(
                self._data, tosparse=True, dtype=dtype, shape=shape)
        else:
            return self

    def tondarray(self):
        """Return a dense numpy.ndarray representation of array.

        Returns
        -------
        arr : numpy.ndarray
            A representation of current data as numpy.ndarray.
            If possible, we avoid copying original data.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray([1,2,3]).tondarray()
        >>> array
        array([1, 2, 3])
        >>> type(array)
        <type 'numpy.ndarray'>

        >>> array = CArray([[1,2],[0,4]],tosparse=True).tondarray()
        >>> array
        array([[1, 2],
               [0, 4]])
        >>> type(array)
        <type 'numpy.ndarray'>

        """
        return self._data.tondarray()

    def tocsr(self):
        """Return a sparse scipy.sparse.csr_matrix representation of array.

        Returns
        -------
        arr : scipy.sparse.csr_matrix
            A representation of current data as scipy.sparse.csr_matrix.
            If possible, we avoid copying original data.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray([[1,2],[0,4]], tosparse=True).tocsr()
        >>> print array  # doctest: +NORMALIZE_WHITESPACE
          (0, 0)	1
          (0, 1)	2
          (1, 1)	4
        >>> type(array)
        <class 'scipy.sparse.csr.csr_matrix'>

        >>> array = CArray([1,2,3]).tocsr()
        >>> print array  # doctest: +NORMALIZE_WHITESPACE
          (0, 0)	1
          (0, 1)	2
          (0, 2)	3
        >>> type(array)
        <class 'scipy.sparse.csr.csr_matrix'>

        """
        return self._data.tocsr()

    def tolist(self):
        """Return the array as a (possibly nested) list.

        Return a copy of the array data as a (nested) Python list.
        Data items are converted to the nearest compatible Python type.

        Returns
        -------
        out : list
            The possibly nested list of array elements.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray([[1,2],[0,4]]).tolist()
        >>> array
        [[1, 2], [0, 4]]
        >>> print CArray(array)
        CArray([[1 2]
         [0 4]])

        >>> array = CArray([[1,2],[0,4]],tosparse=True).tolist()
        >>> array  # integers can be converted to long sometimes  # doctest: +SKIP
        [[1L, 2L], [0L, 4L]]
        >>> print CArray(array,tosparse=True)  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (0, 1)	2
          (1, 1)	4)

        """
        return self._data.tolist()

    # ---------------------------- #
    # # # # # # INDEXING # # # # # #
    # -----------------------------#

    def _check_index(self, idx):
        """Consistency checks for __getitem__ and __setitem__ functions."""
        # This routine only transforms input CArrays into CDense/CSparse
        # Object validity is checked by CDense/CSparse
        if isinstance(idx, tuple):
            # Extracting buffer from CArrays and rebuilding the tuple
            idx_data = tuple(
                dim._data if isinstance(dim, CArray) else dim for dim in idx)

        elif isinstance(idx, CArray):  # CArray list-like
            idx_data = idx._data

        else:  # Nothing to convert
            idx_data = idx

        # Index ready for CArray subclasses
        return idx_data

    def __getitem__(self, idx):
        """Return new array with slicing/indexing result.

        Parameters
        ----------
        idx : object
            - CArray boolean masks
              Number of rows should be equal or higher
              than the number array's dimensions.
            - List of lists (output of `find_2d` method).
              Number of elements should be equal or higher
              than the number array's dimensions.
            - tuple of 2 or more elements. Any of the following:
                - CArray, 1D dense format
                - Iterable built-in types (list, slice).
                - Atomic built-in types (int, bool).
                - Numpy atomic types (np.integer, np.bool_).
            - for vector-like arrays, one element between:
                - Iterable built-in types (list, slice).
                - Atomic built-in types (int, bool).
                - Numpy atomic types (np.integer, np.bool_).

        """
        # Checking consistency and integrity of input index
        idx_data = self._check_index(idx)

        # Calling getitem of data buffer
        return self._instance_array(self._data.__getitem__(idx_data))

    def __setitem__(self, idx, value):
        """Set input data to slicing/indexing result.

        Parameters
        ----------
        idx : object
            - CArray boolean masks
              Number of rows should be equal or higher
              than the number array's dimensions.
            - List of lists (output of `find_2d` method).
              Number of elements should be equal or higher
              than the number array's dimensions.
            - tuple of 2 or more elements. Any of the following:
                - CArray, 1D dense format
                - Iterable built-in types (list, slice).
                - Atomic built-in types (int, bool).
                - Numpy atomic types (np.integer, np.bool_).
            - for vector-like arrays, one element between:
                - Iterable built-in types (list, slice).
                - Atomic built-in types (int, bool).
                - Numpy atomic types (np.integer, np.bool_).
        value : object
            - CArray with shape compatible with index
              For dense format arrays, only dense values can be used.
            - Atomic built-in types (int, float, bool)
            - Numpy atomic types (np.integer, np.floating, np.bool_)

        """
        # Checking consistency and integrity of input index
        idx_data = self._check_index(idx)

        if isinstance(value, CArray):
            # Dense arrays only accept setting dense values
            if self.isdense:
                value = value.todense()
            # Now we extract buffer (CDense, CSparse) from the CArray
            value = value._data

        # Calling setitem of data buffer
        self._data.__setitem__(idx_data, value)

    # ------------------------------------ #
    # # # # # # SYSTEM OVERLOADS # # # # # #
    # -------------------------------------#

    def __add__(self, other):
        """Element-wise addition.

        Parameters
        ----------
        other : CArray or scalar or bool
            Element to add to current array. If a CArray, element-wise
            addition will be performed. If scalar or boolean, the element
            will be sum to each array element.

        Returns
        -------
        array : CArray
            Array after addition.
            If input is a scalar or a boolean, array format is preserved.
            If input is a CArray, format of output array depends on the input:
             - sparse + sparse : sparse
             - sparse + dense : dense
             - dense + sparse : dense
             - dense + dense : dense

        .. warning::
            for sparse format, scalar/bool addition is not available
            due to performance reasons.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__add__(other))
        elif isinstance(other, CArray):
            # dense vs sparse not supported (sparse vs dense IS supported)
            if self.isdense is True and other.issparse is True:
                other = other.todense()
            return self.__class__(self._data.__add__(other._data))
        elif is_ndarray(other) or is_scsarray(other):
            raise TypeError("unsupported operand type(s) for +: "
                            "'{:}' and '{:}'".format(type(self).__name__,
                                                     type(other).__name__))
        else:
            return NotImplemented

    def __radd__(self, other):
        """Element-wise (inverse) addition.

        Parameters
        ----------
        other : scalar or bool
            Element to add to current array.
            The element will be sum to each array element.

        Returns
        -------
        array : CArray
            Array after addition. Format is preserved.

        .. warning::
            for sparse format, scalar/bool addition is not available
            due to performance reasons.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__radd__(other))
        else:
            return NotImplemented

    def __sub__(self, other):
        """Element-wise subtraction.

        Parameters
        ----------
        other : CArray or scalar or bool
            Element to subtract to current array. If a CArray, element-wise
            subtraction will be performed. If scalar or boolean, the element
            will be subtracted to each array element.

        Returns
        -------
        array : CArray
            Array after subtraction.
            If input is a scalar or a boolean, array format is preserved.
            If input is a CArray, format of output array depends on the input:
             - sparse - sparse : sparse
             - sparse - dense : dense
             - dense - sparse : dense
             - dense - dense : dense

        .. warning::
            for sparse format, scalar/bool subtraction is not available
            due to performance reasons.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__sub__(other))
        elif isinstance(other, CArray):
            # dense vs sparse not supported (sparse vs dense IS supported)
            if self.isdense is True and other.issparse is True:
                other = other.todense()
            return self.__class__(self._data.__sub__(other._data))
        elif is_ndarray(other) or is_scsarray(other):
            raise TypeError("unsupported operand type(s) for -: "
                            "'{:}' and '{:}'".format(type(self).__name__,
                                                     type(other).__name__))
        else:
            return NotImplemented

    def __rsub__(self, other):
        """Element-wise (inverse) subtraction.

        Parameters
        ----------
        other : scalar or bool
            Element to subtract to current array.
            The element will be subtracted to each array element.

        Returns
        -------
        array : CArray
            Array after subtraction.

        .. warning::
            for sparse format, scalar/bool subtraction is not available
            due to performance reasons.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__rsub__(other))
        else:
            return NotImplemented

    def __mul__(self, other):
        """Element-wise product.

        Parameters
        ----------
        other : CArray or scalar or bool
            Element to multiply to current array. If a CArray, element-wise
            product will be performed. If scalar or boolean, the element
            will be multiplied to each array element.

        Returns
        -------
        array : CArray
            Array after product.
            If input is a scalar or a boolean, array format is preserved.
            If input is a CArray, format of output array depends on the
            format of current array:
             - sparse * sparse : sparse
             - sparse * dense : sparse
             - dense * sparse : dense
             - dense * dense : dense

        See Also
        --------
        .dot : inner/outer product between arrays.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__mul__(other))
        elif isinstance(other, CArray):
            # dense vs sparse not supported (sparse vs dense IS supported)
            if self.isdense is True and other.issparse is True:
                other = other.todense()
            return self.__class__(self._data.__mul__(other._data))
        elif is_ndarray(other) or is_scsarray(other):
            raise TypeError("unsupported operand type(s) for *: "
                            "'{:}' and '{:}'".format(type(self).__name__,
                                                     type(other).__name__))
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Element-wise (inverse) product.

        Parameters
        ----------
        other : scalar or bool
            Element to multiply to current array.
            The element will be multiplied to each array element.

        Returns
        -------
        array : CArray
            Array after product. Array format is always preserved.

        See Also
        --------
        .dot : inner/outer product between arrays.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__rmul__(other))
        else:
            return NotImplemented

    def __truediv__(self, other):
        """Element-wise true division.

        Parameters
        ----------
        other : CArray or scalar or bool
            Element to divide to current array. If a CArray, element-wise
            division will be performed. If scalar or boolean, the element
            will be divided to each array element.

        Returns
        -------
        array : CArray
            Array after division.
            If input is a scalar or a boolean, array format is preserved.
            If input is a CArray, format of output array depends on the
            format of current array:
             - sparse / sparse : sparse
             - sparse / dense : sparse
             - dense / sparse : dense
             - dense / dense : dense

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__div__(other))
        elif isinstance(other, CArray):
            # dense vs sparse not supported (sparse vs dense IS supported)
            if self.isdense is True and other.issparse is True:
                other = other.todense()
            return self.__class__(self._data.__div__(other._data))
        elif is_ndarray(other) or is_scsarray(other):
            raise TypeError("unsupported operand type(s) for /: "
                            "'{:}' and '{:}'".format(type(self).__name__,
                                                     type(other).__name__))
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """Element-wise (inverse) true division.

        Parameters
        ----------
        other : CArray or scalar or bool
            Element to divide to current array. If a CArray, element-wise
            division will be performed. If scalar or boolean, the element
            will be divided to each array element.

        Returns
        -------
        array : CArray
            Array after division. Array format is always preserved.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__rtruediv__(other))
        else:
            return NotImplemented

    def __div__(self, other):
        """Element-wise division.

        See .__truediv__() for more informations.

        """
        return self.__truediv__(other)

    def __rdiv__(self, other):
        """Element-wise (inverse) division.

        See .__rtruediv__() for more informations.

        """
        return self.__rtruediv__(other)

    def __floordiv__(self, other):
        """Element-wise floor division (// operator).

        Parameters
        ----------
        other : CArray or scalar or bool
            Element to divide to current array. If a CArray, element-wise
            division will be performed. If scalar or boolean, the element
            will be divided to each array element.

        Returns
        -------
        array : CArray
            Array after division.
            If input is a scalar or a boolean, array format is preserved.
            If input is a CArray, format of output array depends on the
            format of current array:
             - sparse / sparse : sparse
             - sparse / dense : sparse
             - dense / sparse : dense
             - dense / dense : dense

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__floordiv__(other))
        elif isinstance(other, CArray):
            # dense vs sparse not supported (sparse vs dense IS supported)
            if self.isdense is True and other.issparse is True:
                other = other.todense()
            return self.__class__(self._data.__floordiv__(other._data))
        elif is_ndarray(other) or is_scsarray(other):
            raise TypeError("unsupported operand type(s) for //: "
                            "'{:}' and '{:}'".format(type(self).__name__,
                                                     type(other).__name__))
        else:
            return NotImplemented

    def __rfloordiv__(self, other):
        """Element-wise (inverse) floor division (// operator).

        Parameters
        ----------
        other : CArray or scalar or bool
            Element to divide to current array. If a CArray, element-wise
            division will be performed. If scalar or boolean, the element
            will be divided to each array element.

        Returns
        -------
        array : CArray
            Array after division. Array format is always preserved.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__rfloordiv__(other))
        else:
            return NotImplemented

    def __abs__(self):
        """Returns array elements without sign.

        Returns
        -------
        array : CArray
            Array with the corresponding elements without sign.

        """
        return self.__class__(self._data.__abs__())

    def __neg__(self):
        """Returns array elements with negated sign.

        Returns
        -------
        array : CArray
            Array with the corresponding elements with negated sign.

        """
        return self.__class__(self._data.__neg__())

    def __pow__(self, power):
        """Element-wise power.

        Parameters
        ----------
        power : CArray or scalar or bool
            Power to use. If scalar or boolean, each array element will be
            elevated to power. If a CArray, each array element will be
            elevated to the corresponding element of the input array.

        Returns
        -------
        array : CArray
            Array after power. Array format is always preserved.

        .. warning:: sparse ** array is not supported.

        """
        if is_scalar(power) or is_bool(power):
            return self.__class__(self._data.__pow__(power))
        elif isinstance(power, CArray):
            # dense vs sparse not supported (sparse vs dense IS supported)
            if self.isdense is True and power.issparse is True:
                power = power.todense()
            return self.__class__(self._data.__pow__(power._data))
        elif is_ndarray(power) or is_scsarray(power):
            raise TypeError("unsupported operand type(s) for **: "
                            "'{:}' and '{:}'".format(type(self).__name__,
                                                     type(power).__name__))
        else:
            return NotImplemented

    def __rpow__(self, power):
        """Element-wise (inverse) power.

        DENSE FORMAT ONLY

        Parameters
        ----------
        power : scalar or bool
            Power to use. Each array element will be elevated to power.

        Returns
        -------
        array : CArray
            Array after power. Array format is always preserved.

        """
        if is_scalar(power) or is_bool(power):
            return self.__class__(self._data.__rpow__(power))
        else:
            return NotImplemented

    def __eq__(self, other):
        """Element-wise == operator.

        Parameters
        ----------
        other : CArray or scalar or bool
            Element to be compared. If a CArray, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CArray
            Boolean array with comparison result.
            If input is a scalar or a boolean, array format is preserved.
            If input is a CArray, format of output array depends on the input:
             - sparse == sparse : sparse
             - sparse == dense : dense
             - dense == sparse : dense
             - dense == dense : dense

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__eq__(other))
        elif isinstance(other, CArray):
            # dense vs sparse not supported (sparse vs dense IS supported)
            if self.isdense is True and other.issparse is True:
                other = other.todense()
            return self.__class__(self._data.__eq__(other._data))
        elif is_ndarray(other) or is_scsarray(other):
            raise TypeError("unsupported operand type(s) for ==: "
                            "'{:}' and '{:}'".format(type(self).__name__,
                                                     type(other).__name__))
        else:  # Any unmanaged object is considered not-equal
            return False

    def __lt__(self, other):
        """Element-wise < operator.

        Parameters
        ----------
        other : CArray or scalar or bool
            Element to be compared. If a CArray, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CArray
            Boolean array with comparison result.
            If input is a scalar or a boolean, array format is preserved.
            If input is a CArray, format of output array depends on the input:
             - sparse < sparse : sparse
             - sparse < dense : dense
             - dense < sparse : dense
             - dense < dense : dense

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__lt__(other))
        elif isinstance(other, CArray):
            # dense vs sparse not supported (sparse vs dense IS supported)
            if self.isdense is True and other.issparse is True:
                other = other.todense()
            return self.__class__(self._data.__lt__(other._data))
        else:
            raise TypeError("unsupported operand type(s) for <: "
                            "'{:}' and '{:}'".format(type(self).__name__,
                                                     type(other).__name__))

    def __le__(self, other):
        """Element-wise <= operator.

        Parameters
        ----------
        other : CArray or scalar or bool
            Element to be compared. If a CArray, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CArray
            Boolean array with comparison result.
            If input is a scalar or a boolean, array format is preserved.
            If input is a CArray, format of output array depends on the input:
             - sparse <= sparse : sparse
             - sparse <= dense : dense
             - dense <= sparse : dense
             - dense <= dense : dense

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__le__(other))
        elif isinstance(other, CArray):
            # dense vs sparse not supported (sparse vs dense IS supported)
            if self.isdense is True and other.issparse is True:
                other = other.todense()
            return self.__class__(self._data.__le__(other._data))
        else:
            raise TypeError("unsupported operand type(s) for <=: "
                            "'{:}' and '{:}'".format(type(self).__name__,
                                                     type(other).__name__))

    def __gt__(self, other):
        """Element-wise > operator.

        Parameters
        ----------
        other : CArray or scalar or bool
            Element to be compared. If a CArray, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CArray
            Boolean array with comparison result.
            If input is a scalar or a boolean, array format is preserved.
            If input is a CArray, format of output array depends on the input:
             - sparse > sparse : sparse
             - sparse > dense : dense
             - dense > sparse : dense
             - dense > dense : dense

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__gt__(other))
        elif isinstance(other, CArray):
            # dense vs sparse not supported (sparse vs dense IS supported)
            if self.isdense is True and other.issparse is True:
                other = other.todense()
            return self.__class__(self._data.__gt__(other._data))
        else:
            raise TypeError("unsupported operand type(s) for >: "
                            "'{:}' and '{:}'".format(type(self).__name__,
                                                     type(other).__name__))

    def __ge__(self, other):
        """Element-wise >= operator.

        Parameters
        ----------
        other : CArray or scalar or bool
            Element to be compared. If a CArray, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CArray
            Boolean array with comparison result.
            If input is a scalar or a boolean, array format is preserved.
            If input is a CArray, format of output array depends on the input:
             - sparse >= sparse : sparse
             - sparse >= dense : dense
             - dense >= sparse : dense
             - dense >= dense : dense

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__ge__(other))
        elif isinstance(other, CArray):
            # dense vs sparse not supported (sparse vs dense IS supported)
            if self.isdense is True and other.issparse is True:
                other = other.todense()
            return self.__class__(self._data.__ge__(other._data))
        else:
            raise TypeError("unsupported operand type(s) for >=: "
                            "'{:}' and '{:}'".format(type(self).__name__,
                                                     type(other).__name__))

    def __ne__(self, other):
        """Element-wise != operator.

        Parameters
        ----------
        other : CArray or scalar or bool
            Element to be compared. If a CArray, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CArray
            Boolean array with comparison result.
            If input is a scalar or a boolean, array format is preserved.
            If input is a CArray, format of output array depends on the input:
             - sparse != sparse : sparse
             - sparse != dense : dense
             - dense != sparse : dense
             - dense != dense : dense

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__ne__(other))
        elif isinstance(other, CArray):
            # dense vs sparse not supported (sparse vs dense IS supported)
            if self.isdense is True and other.issparse is True:
                other = other.todense()
            return self.__class__(self._data.__ne__(other._data))
        elif is_ndarray(other) or is_scsarray(other):
            raise TypeError("unsupported operand type(s) for !=: "
                            "'{:}' and '{:}'".format(type(self).__name__,
                                                     type(other).__name__))
        else:  # Any unmanaged object is considered not-equal
            return True

    def __bool__(self):
        """Manage 'and' and 'or' operators."""
        return bool(self._data)

    __nonzero__ = __bool__  # Compatibility with python < 3

    def __iter__(self):
        """Yields array elements in raster-scan order.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray([[1, 2], [3, 4]])
        >>> for elem in array:
        ...     print elem
        1
        2
        3
        4

        >>> array = CArray([1, 2, 3, 4])
        >>> for elem in array:
        ...     print elem
        1
        2
        3
        4

        """
        # The following can be simplified by ravelling the array first
        # But as .ravel() can return a copy, we prefer this
        n_rows = 1 if self.is_vector_like else self.shape[0]
        n_columns = self.size if self.is_vector_like else self.shape[1]
        for row_id in xrange(n_rows):
            for column_id in xrange(n_columns):
                yield self[row_id, column_id]

    def __str__(self):
        """Define `print` (or `str`) behaviour.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([1])
        CArray([1])
        >>> print CArray([1,2])
        CArray([1 2])
        >>> print CArray([[1,2],[3,4]])
        CArray([[1 2]
         [3 4]])
        >>> print CArray([[1,2],[3,4]], tosparse=True)  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (0, 1)	2
          (1, 0)	3
          (1, 1)	4)

        """
        # Storing numpy format settings
        np_format = np.get_printoptions()
        # Preventing newlines and toggling summarization as often as possible
        np.set_printoptions(
            threshold=36, linewidth=79, edgeitems=3, precision=6)
        # Build the string
        a_str = self.__class__.__name__ + "(" + str(self._data) + ")"
        # Restoring numpy format settings
        np.set_printoptions(**np_format)
        return a_str

    def __repr__(self):
        """Define `repr` behaviour of array.

        Examples
        --------
        >>> from secml.array import CArray

        >>> repr(CArray([]))
        'CArray(0,)(dense: [])'
        >>> repr(CArray([1]))
        'CArray(1,)(dense: [1])'
        >>> repr(CArray([1,2]))
        'CArray(2,)(dense: [1 2])'
        >>> repr(CArray([1,2,3]))
        'CArray(3,)(dense: [1 2 3])'
        >>> repr(CArray([[1,2]]))
        'CArray(1, 2)(dense: [[1 2]])'
        >>> repr(CArray([[1,2,3]]))
        'CArray(1, 3)(dense: [[1 2 3]])'
        >>> repr(CArray([[1,2],[3,4]]))
        'CArray(2, 2)(dense: [[1 2] [3 4]])'
        >>> repr(CArray([[1,2,3],[4,5,6]]))
        'CArray(2, 3)(dense: [[1 2 3] [4 5 6]])'
        >>> repr(CArray([[1,2,3],[4,5,6],[7,8,9]]))
        'CArray(3, 3)(dense: [[1 2 3] [4 5 6] [7 8 9]])'
        >>> repr(CArray.randint(10, shape=(3,10)))  # doctest: +SKIP
        'CArray(3, 10)(dense: [[2 9 3 ..., 6 1 1] [2 7 5 ..., 9 5 5] [9 6 4 ..., 8 7 5]])'
        >>> repr(CArray([[1,2],[3,4]], tosparse=True))  # doctest: +NORMALIZE_WHITESPACE
        'CArray(2, 2)(sparse: (0, 0) 1  (0, 1) 2  (1, 0) 3  (1, 1) 4)'

        """
        import re
        # Storing numpy format settings
        np_format = np.get_printoptions()
        # Preventing newlines and toggling summarization as often as possible
        np.set_printoptions(
            threshold=36, linewidth=79, edgeitems=3, precision=6)
        # Starting with CArray(shape)...
        repr_str = self.__class__.__name__ + str(self.shape)
        # Replace any line separator
        array_repr = re.sub(r'\r|\n', '', str(self._data))
        if self.isdense is True:
            repr_str += '(dense: ' + array_repr
        elif self.issparse is True:
            repr_str += '(sparse: '
            # Replace any tabuler char
            repr_str += re.sub(r'\t', ' ', array_repr[2:])
        # Restoring numpy format settings
        np.set_printoptions(**np_format)
        return repr_str + ')'

    # ------------------------------ #
    # # # # # # COPY UTILS # # # # # #
    # -------------------------------#

    def __copy__(self):
        """Called when copy.copy(CArray) is called."""
        return self.__class__(self)

    def deepcopy(self):
        """Return a deepcopy of current array.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray([1,2,3])
        >>> array_deepcopy = array.deepcopy()
        >>> array_deepcopy[0] = 100

        >>> print array
        CArray([1 2 3])
        >>> print array_deepcopy
        CArray([100   2   3])

        """
        return deepcopy(self)

    def __deepcopy__(self, memo):
        """Called when copy.deepcopy(CArray) is called."""
        return self.__class__(deepcopy(self._data, memo))

    # ----------------------------- #
    # # # # # # SAVE/LOAD # # # # # #
    # ------------------------------#

    def save(self, datafile, overwrite=False):
        """Save array data into plain text file.

        Data is stored preserving original data type.

        Parameters
        ----------
        datafile : str, file_handle (dense only)
            Text file to save data to. If a string, it's supposed
            to be the filename of file to save. If file is a file
            handle, data will be stored using active file handle mode.
            If the filename ends in .gz, the file is automatically
            saved in compressed gzip format. load() function understands
            gzipped files transparently.
        overwrite : bool, optional
            If True and target file already exists, file will be overwritten.
            Otherwise (default), IOError will be raised.

        Notes
        -----
            - Dense format, flat arrays are stored with shape N x 1.
            - Sparse format, we only save non-zero data along with indices
                necessary to reconstruct original 2-dimensional array.
            - Dense format, shape of original array can be easly recognized
                from target text file.

        """
        if self.issparse is True and not isinstance(datafile, str):
            # TODO: WE CAN ALLOW FILE HANDLE SAVING?!
            raise NotImplementedError(
                "Save using file handle is only supported for dense arrays.")
        else:
            self._data.save(datafile, overwrite=overwrite)

    @classmethod
    def load(cls, datafile, arrayformat='dense', dtype=float,
             startrow=0, skipend=0, cols=None):
        """Load array data from plain text file.

        Parameters
        ----------
        datafile : str, file_handle
            File or filename to read. If the filename extension
            is gz or bz2, the file is first decompressed.
        arrayformat : {'dense', 'sparse'}, optional
            Format of array to load, default 'dense'.
        dtype : str, dtype, optional
            Data type of the resulting array, default 'float'. If None,
            the dtype will be determined by the contents of the file.
        startrow : int, optional, dense only
            Array row to start loading from.
        skipend : int, optional, dense only
            Number of lines to skip from the end of the file when reading.
        cols : {CArray, int, tuple, slice}, optional, dense only
            Columns to load from target file.

        Returns
        -------
        loaded : CArray
            Array resulting from loading, 2-dimensional.

        """
        if arrayformat is 'dense':
            if cols is None:
                cols = CArray([])
            return cls(CDense.load(datafile, dtype=dtype, startrow=startrow,
                                   skipend=skipend, cols=cols._data))
        elif arrayformat is 'sparse':
            return cls(CSparse.load(datafile, dtype=dtype))
        else:
            raise ValueError("Supported arrayformat are 'dense' and 'sparse'.")

    # ----------------------------- #
    # # # # # # UTILITIES # # # # # #
    # ------------------------------#

    def has_compatible_shape(self, other):
        """Return True if input CArray has a compatible shape.

        Two CArrays can be considered compatible if
        both have the same shape or both are vector-like.

        Parameters
        ----------
        other : CArray
            Array to check for shape compatibility

        Return
        ------
        out_check : bool
            True if input array has compatible shape with
            current array.

        See Also
        --------
        .is_vector_like : check if an array is vector-like.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[1,2]]).has_compatible_shape(CArray([[1],[2]]))
        False

        >>> print CArray([1,2]).has_compatible_shape(CArray([1,2,3]))
        False

        >>> print CArray([[1,2]], tosparse=True).has_compatible_shape(CArray([1,2]))
        True

        """
        if self.shape == other.shape:  # Standard case: same shape
            return True
        # 2 flats, 1-flat and 1 with (1, x) shape, 2 with (1, x) shape
        elif self.is_vector_like is True:
            return other.is_vector_like is True and self.size == other.size
        return False

    def transpose(self):
        """Returns current array with axes transposed.
        A view is returned if possible.

        Returns
        -------
        out : CArray
            A view, if possibile, of current array with axes suitably permuted.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([1, 2, 3]).transpose()
        CArray([[1]
         [2]
         [3]])

        >>> print CArray([[1], [2], [3]]).transpose()
        CArray([[1 2 3]])

        """
        return self.__class__(self._data.transpose())

    def unique(self, return_index=False, return_inverse=False):
        """Find the unique elements of an array.

        There are two optional outputs in addition to the unique
        elements: the indices of the input array that give the
        unique values, and the indices of the unique array that
        reconstruct the input array.

        Parameters
        ----------
        return_index : bool, optional, dense only
            If True, also return the indices of array that result
            in the unique array (default False).
        return_inverse : bool, optional, dense only
            If True, also return the indices of the unique array
            that can be used to reconstruct the original array
            (default False).

        Returns
        -------
        unique : CArray
            Dense array with the sorted unique values of the array.
        unique_indices : CArray, optional, dense only
            The indices of the first occurrences of the unique values
            in the (flattened) original array. Only provided if
            return_index is True.
        unique_inverse : CArray, optional, dense only
            The indices to reconstruct the (flattened) original array
            from the unique array. Only provided if return_inverse is True.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[1,0,2],[2,0,3]]).unique()
        CArray([0 1 2 3])

        >>> print CArray([1,2,2,3,3], tosparse=True).unique()
        CArray([1 2 3])

        >>> u, u_idx, u_inv = CArray([[2,2,3,3]]).unique(return_index=True, return_inverse=True)
        >>> print u, u_idx  # unique and unique_indices
        CArray([2 3]) CArray([0 2])
        >>> print u[u_inv]  # original (flattened) array reconstructed from unique_inverse
        CArray([2 2 3 3])

        """
        if self.isdense is True:
            out = self._data.unique(return_index, return_inverse)
            if isinstance(out, tuple):  # unique returned multiple elements
                return tuple([self.__class__(elem) for elem in out])
            else:
                return self.__class__(out)
        else:
            if return_index is not False or return_inverse is not False:
                raise ValueError("return_index and return_inverse parameters "
                                 "are only available for dense arrays.")
            return self.__class__(self._data.unique())

    def append(self, array2, axis=None):
        """Append values to the end of an array.

        Parameters
        ----------
        array2 : CArray or array_like
            Second array. If array2 is not an array, a CArray will be
            created before appending.
        axis : int or None, optional
            The axis along which values are appended.
            If axis is None, both arrays are flattened before use.

        Returns
        -------
        out_append : CArray
            A copy of array with values appended to axis. Note that append
            does not occur in-place: a new array is allocated and filled.
            If axis is None, out is a flattened array. Always return an
            array with the same format of original array.

        Notes
        -----
        Differently from numpy, we manage flat vectors as 2-Dimensional of
        shape (1, array.size). Consequently, result of appending a flat
        array to a flat array is 1-Dimensional only if axis=1. Appending
        a flat array to a 2-Dimensional array, or viceversa, always results
        in a 2-Dimensional array.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[1,2],[3,4]]).append([[11],[22]])
        CArray([ 1  2  3  4 11 22])
        >>> print CArray([[1,2],[3,4]]).append([[11,22]], axis=0)
        CArray([[ 1  2]
         [ 3  4]
         [11 22]])

        >>> print CArray([[1,2],[3,4]]).append(CArray([[11],[22]], tosparse=True))
        CArray([ 1  2  3  4 11 22])
        >>> array = CArray([[1,2],[3,4]], tosparse=True).append([[11],[22]])
        >>> print array  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (0, 1)	2
          (0, 2)	3
          (0, 3)	4
          (0, 4)	11
          (0, 5)	22)

        >>> print CArray([1,2]).append([11,22])
        CArray([ 1  2 11 22])

        >>> print CArray([1,2]).append([11,22], axis=0)
        CArray([[ 1  2]
         [11 22]])
        >>> print CArray([1,2]).append([11,22], axis=1)
        CArray([ 1  2 11 22])

        """
        if self.issparse:  # Return sparse if first array is sparse
            array2 = CArray(array2, tosparse=True)
        else:
            array2 = CArray(array2).todense()

        return self.__class__(self._data.append(array2._data, axis=axis))

    def dot(self, x):
        """Dot product of two arrays.

        For 2-dim arrays it is equivalent to matrix multiplication.
        If both arrays are dense flat (rows), it is equivalent to the
        inner product of vectors (without complex conjugation).

        Format of output array is the same of the first product argument.

        Parameters
        ----------
        x : CArray
            Second argument of dot product.

        Returns
        -------
        out : CArray, scalar_like
            Result of dot product.
            A CArray with the same format of first argument or
            scalar if out.size == 1.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[1,1],[2,2]]).dot(CArray([[1,1],[0,0]], tosparse=True))
        CArray([[1 1]
         [2 2]])

        >>> print CArray([10,20]).dot(CArray([[1],[0]], tosparse=True))
        10

        # OUTER PRODUCT
        >>> print CArray([[10],[20]]).dot(CArray([1,0], tosparse=True))
        CArray([[10  0]
         [20  0]])

        # INNER PRODUCT BETWEEN VECTORS
        >>> print CArray([10,20]).dot(CArray([1,0]))
        10

        # Inner product between vector-like arrays is a matrix multiplication
        >>> print CArray([10,20]).dot(CArray([1,0], tosparse=True).T)
        10
        >>> print CArray([10,20], tosparse=True).dot(CArray([1,0]).T)
        10

        """
        # We have to handle only one problematic case: dense vs sparse dot
        if self.isdense is True and x.issparse is True:
            return self._instance_array(self._data.dot(x.todense()._data))
        else:
            return self._instance_array(self._data.dot(x._data))

    def reshape(self, newshape):
        """Gives a new shape to an array without changing its data.

        Parameters
        ----------
        newshape : int or sequence of ints
            Desired shape for output array.
            If an integer or a tuple of length 1, resulting array
            will have shape (n,) if dense, (1,n) if sparse.

        Returns
        -------
        out : CArray
            Array with new shape. If possible, a view of original array data
            will be returned, otherwise a copy will be made first.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([1,2,3]).reshape((3,1))
        CArray([[1]
         [2]
         [3]])

        >>> print CArray([[1],[2],[3]], tosparse=True).reshape((3,))  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (0, 1)	2
          (0, 2)	3)

        >>> CArray([1,2,3]).reshape(4)
        Traceback (most recent call last):
          ...
        ValueError: cannot reshape array of size 3 into shape (4,)

        """
        return self.__class__(self._data.reshape(newshape))

    def resize(self, newshape, constant=0):
        """Return a new array with the specified shape.

        Missing entries are filled with input constant (default 0).

        DENSE FORMAT ONLY

        Parameters
        ----------
        newshape : int or sequence of ints
            Integer or one integer for each desired dimension of output array.
            If a tuple of length 1, output sparse array will have shape (1, n).
        constant : scalar
            Scalar to be used for filling missing entries. Default 0.

        Returns
        -------
        out : CArray
            Array with new shape. Array dtype is preserved.
            Missing entries are filled with the desired constant (default 0).

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([1,2,3]).resize((3,3))
        CArray([[1 2 3]
         [0 0 0]
         [0 0 0]])

        >>> print CArray([1,2,3]).resize((3,1))
        CArray([[1]
         [2]
         [3]])

        >>> print CArray([1,2,3]).resize((1,3))
        CArray([[1 2 3]])

        >>> print CArray([[1,2,3]]).resize((5, ))
        CArray([1 2 3 0 0])

        >>> from secml.core.constants import inf
        >>> print CArray([[1,2,3]]).resize((5, ), constant=inf)  # doctest: +SKIP
        CArray([                   1                    2                    3
         -9223372036854775808 -9223372036854775808])

        >>> print CArray([[0, 1],[2, 3]]).resize(3)
        CArray([0 1 2])

        >>> print CArray([[0, 1],[2, 3]]).resize((3, 3))
        CArray([[0 1 2]
         [3 0 0]
         [0 0 0]])

        >>> print CArray([[0, 1, 2],[3, 4, 5]]).resize((2, 2))
        CArray([[0 1]
         [2 3]])

        """
        return self.__class__(self._data.resize(newshape, constant=constant))

    def astype(self, dtype):
        """Copy of the array, casted to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.

        Returns
        -------
        out : CArray
            Copy of the original array casted to new data type.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([1, 2, 3]).astype(float)
        CArray([ 1.  2.  3.])

        >>> print CArray([1.1, 2.1, 3.1], tosparse=True).astype(int)  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (0, 1)	2
          (0, 2)	3)

        """
        return self.__class__(self._data.astype(dtype))

    def diag(self, k=0):
        """Extract a diagonal from array or construct a diagonal array.

        Parameters
        ----------
        k : int
            Diagonal index. Default is 0.
            Use k > 0 for diagonals above the main diagonal,
            k < 0 for diagonals below the main diagonal.
            This parameter used to extract the k-th diagonal
            is not supported for sparse arrays.

        Returns
        -------
        diag : CArray
            The extracted diagonal or constructed diagonal array.
            If array is 2-Dimensional, returns its k-th diagonal.
             Depending on numpy version resulting array can be read-only
             or a view of the original array's diagonal. To make output
             array writable, use deepcopy().
            If array is vector_like, return a 2-D array with
             the array on the k-th diagonal.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[1, 2, 3], [10, 20, 30]]).diag(k=1)
        CArray([ 2 30])

        >>> print CArray([[1, 1]], tosparse=True).diag()  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (1, 1)	1)

        >>> print CArray([[1, 1], [0, 0]], tosparse=True).diag(k=1)
        Traceback (most recent call last):
          ...
        ValueError: Extracting the k-th diagonal is not supported.

        """
        if self.size == 0:
            raise ValueError("cannot use diag() on empty arrays.")

        # Avoid extracting diagonal of a 2-D dense array with shape[0] == 1
        data = self._data.ravel() if self.is_vector_like else self._data

        return self.__class__(data.diag(k=k))

    def argmax(self, axis=None):
        """Indices of the maximum values along an axis.

        Parameters
        ----------
        axis : int, None, optional
            If None (default), array is flattened before computing
            index, otherwise the specified axis is used.

        Returns
        -------
        index : int, CArray
            Scalar with index of the maximum value for flattened array or
            Carray with indices along the given axis.

        Notes
        -----
        In case of multiple occurrences of the maximum values, the
        indices corresponding to the first occurrence are returned.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([-1, 0, 3]).argmax()
        2

        >>> print CArray([[-1, 0],[4, 3]]).argmax(axis=0)  # We return the index of maximum for each column
        CArray([[1 1]])
        
        >>> print CArray([[-1, 0],[4, 3]]).argmax(axis=1)  # We return the index of maximum for each row
        CArray([[1]
         [0]])

        >>> print CArray([-3,0,1,2]).argmax(axis=0)
        CArray([0 0 0 0])
        >>> print CArray([-3,0,1,2]).argmax(axis=1)
        3

        """
        return self._instance_array(self._data.argmax(axis=axis))

    def argmin(self, axis=None):
        """Indices of the minimum values along an axis.

        Parameters
        ----------
        axis : int, None, optional
            If None (default), array is flattened before computing
            index, otherwise the specified axis is used.

        Returns
        -------
        index : int, CArray
            Scalar with index of the minimum value for flattened array or
            Carray with indices along the given axis.

        Notes
        -----
        In case of multiple occurrences of the minimum values, the
        indices corresponding to the first occurrence are returned.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([-1, 0, 3]).argmin()
        0

        >>> print CArray([[-1, 0],[4, 3]]).argmin(axis=0)  # We return the index of minimum for each column
        CArray([[0 0]])

        >>> print CArray([[-1, 0],[4, 3]]).argmin(axis=1)  # We return the index of maximum for each row
        CArray([[0]
         [1]])

        >>> print CArray([-3,0,1,2]).argmin(axis=0)
        CArray([0 0 0 0])
        >>> print CArray([-3,0,1,2]).argmin(axis=1)
        0

        """
        return self._instance_array(self._data.argmin(axis=axis))

    def nanargmax(self, axis=None):
        """Indices of the maximum values along an axis ignoring Nans.

        For all-NaN slices ValueError is raised.
        Warning: the results cannot be trusted if a slice
        contains only NaNs and Infs.

        DENSE ARRAYS ONLY

        Parameters
        ----------
        axis : int, None, optional
            If None (default), array is flattened before computing
            index, otherwise the specified axis is used.

        Returns
        -------
        index : int, CArray
            Scalar with index of the maximum value for flattened array or
            CArray with indices along the given axis.

        Notes
        -----
        In case of multiple occurrences of the maximum values, the
        indices corresponding to the first occurrence are returned.

        Examples
        --------
        >>> from secml.array import CArray

        >>> import numpy as np
        >>> print CArray([5, np.nan]).argmax()
        1

        >>> print CArray([5, np.nan]).nanargmax()
        0

        >>> print CArray([[-1, np.nan], [np.nan, 0]]).nanargmax()
        3

        >>> print CArray([[-1, np.nan], [np.nan, 0]]).nanargmax(axis=0)
        CArray([[0 1]])
        >>> print CArray([[-1, np.nan], [np.nan, 0]]).nanargmax(axis=1)
        CArray([[0]
         [1]])

        """
        return self._instance_array(self._data.nanargmax(axis=axis))

    def nanargmin(self, axis=None):
        """Indices of the minimum values along an axis ignoring Nans

        For all-NaN slices ValueError is raised.
        Warning: the results cannot be trusted if a slice
        contains only NaNs and Infs.

        Parameters
        ----------
        axis : int, None, optional
            If None (default), array is flattened before computing
            index, otherwise the specified axis is used.

        Returns
        -------
        index : int, CArray
            Scalar with index of the minimum value for flattened array or
            CArray with indices along the given axis.

        Notes
        -----
        In case of multiple occurrences of the minimum values, the
        indices corresponding to the first occurrence are returned.

        Examples
        --------
        >>> from secml.array import CArray

        >>> import numpy as np
        >>> print CArray([5, np.nan]).argmin()
        1

        >>> print CArray([5, np.nan]).nanargmin()
        0

        >>> print CArray([[-1, np.nan], [np.nan, 0]]).nanargmin()
        0

        >>> print CArray([[-1, np.nan], [np.nan, 0]]).nanargmin(axis=0)
        CArray([[0 1]])
        >>> print CArray([[-1, np.nan], [np.nan, 0]]).nanargmin(axis=1)
        CArray([[0]
         [1]])

        """
        return self._instance_array(self._data.nanargmin(axis=axis))

    def sum(self, axis=None, keepdims=True):
        """Sum of array elements over a given axis.

        Parameters
        ----------
        axis : int, None, optional
            Axis along which a sum is performed. The default
            (axis = None) is perform a sum over all the
            dimensions of the input array. axis may be negative,
            in which case it counts from the last to the first axis.
        keepdims : bool, optional, dense only
            If this is set to True (default), the result will
            broadcast correctly against the original array.
            Otherwise resulting array is flattened.

        Returns
        -------
        out : scalar_like, CArray
            A 2-dim array with the elements sum along specified axis.
            If axis is None, a scalar is returned.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([-3,0,2]).sum()
        -1

        >>> print CArray([[-3,0],[1,2]], tosparse=True).sum(axis=1)
        CArray([[-3]
         [ 3]])

        >>> print CArray([-3,0,1,2]).sum(axis=0)
        CArray([-3  0  1  2])
        >>> print CArray([-3,0,1,2]).sum(axis=1)
        0

        """
        return self._instance_array(
            self._data.sum(axis=axis, keepdims=keepdims))

    def cumsum(self, axis=None):
        """Return the cumulative sum of the array elements along a given axis.

        DENSE FORMAT ONLY

        Parameters
        ----------
        axis : int, None, optional
            Axis along which the cumulative sum is computed.
            The default (None) is to compute the cumsum over
            the flattened array.

        Returns
        -------
        out : CArray
            New array with cumulative sum of elements.
            If axis is None, flat array with same size of input array.
            If axis is not None, same shape of input array.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([-3,0,2]).cumsum()
        CArray([-3 -3 -1])

        >>> print CArray([-3,0,1,2]).cumsum(axis=0)
        CArray([-3  0  1  2])
        >>> print CArray([-3,0,1,2]).cumsum(axis=1)
        CArray([-3 -3 -2  0])

        >>> print CArray([[-3,0],[1,2]]).cumsum()
        CArray([-3 -3 -2  0])

        >>> print CArray([[-3,0],[1,2]]).cumsum(axis=1)
        CArray([[-3 -3]
         [ 1  3]])

        """
        return self.__class__(self._data.cumsum(axis=axis))

    def prod(self, axis=None, dtype=None, keepdims=True):
        """Return the product of array elements over a given axis.

        Parameters
        ----------
        axis : int or None, optional
            Axis along which the product is computed. The default (None)
            is to compute the product over the flattened array.
        dtype : str or dtype, optional
            The data-type of the returned array, as well as of the
            accumulator in which the elements are multiplied. By default,
            if array is of integer type, dtype is the default platform
            integer. (Note: if the type of array is unsigned, then so is
            dtype.) Otherwise, the dtype is the same as that of array.
        keepdims : bool, optional
            If this is set to True (default), the result will
            broadcast correctly against the original array.
            Otherwise resulting array is flattened.

        Returns
        -------
        out_prod : scalar or CArray
            A 2-dim array with the elements product along specified axis.
            If axis is None, a scalar is returned.

        Notes
        -----
        Differently from numpy, we manage flat vectors as 2-Dimensional of
        shape (1, array.size). This means that when axis=0, a flat array is
        returned as is (see examples).

        Arithmetic is modular when using integer types, and no error is
        raised on overflow. That means that, on a 32-bit platform:

        >>> print CArray([536870910, 536870910, 536870910, 536870910]).prod()  # random result  # doctest: +SKIP
        16

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[1,2],[3,4]]).prod()
        24

        >>> print CArray([[1,2],[3,4]], tosparse=True).prod(axis=1)  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	2
          (1, 0)	12)
        >>> print CArray([[1,2],[3,4]]).prod(axis=0, dtype=float)
        CArray([[ 3.  8.]])

        >>> print CArray([1,2,3]).prod(axis=0)
        CArray([1 2 3])
        >>> print CArray([1,2,3]).prod(axis=1)
        6

        """
        return self._instance_array(
            self._data.prod(axis=axis, dtype=dtype, keepdims=keepdims))

    def clip(self, c_min, c_max):
        """Clip (limit) the values in an array.

        DENSE FORMAT ONLY

        Given an interval, values outside the interval are clipped
        to the interval edges. For example, if an interval of [0, 1]
        is specified, values smaller than 0 become 0, and values
        larger than 1 become 1.

        Parameters
        ----------
        c_min, c_max : int
            Clipping intervals.

        Returns
        -------
        out_clip : CArray
            Returns a new array containing the clipped array elements.
            Dtype of the output array depends on the dtype of original array
            and on the dtype of the clipping limits.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[1,2],[3,4]]).clip(2, 4)
        CArray([[2 2]
         [3 4]])

        >>> from secml.core.constants import inf

        >>> # inf is a float, array will be casted accordingly
        >>> print CArray([[1,2],[3,4]]).clip(-inf, 2)
        CArray([[ 1.  2.]
         [ 2.  2.]])

        """
        if c_min > c_max:
            raise ValueError("c_min ({:}) must be lower than "
                             "c_max ({:})".format(c_min, c_max))
        return self.__class__(self._data.clip(c_min, c_max))

    def find(self, condition):
        """Returns vector-like array elements indices depending on condition.

        Parameters
        ----------
        condition : CArray or array_like
            Array like with booleans representing desired condition.

        Returns
        -------
        out_find : list
            List with indices corresponding to array elements
            where condition is True.

        See Also
        --------
        .find_2d : find method for arrays of generic shape.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray([1,0,-6,2,0])
        >>> array_find = array.find(array > 0)
        >>> print array_find
        [0, 3]
        >>> print array[array_find]
        CArray([1 2])

        >>> array = CArray([[1,0,-6,2,0]])
        >>> array_find = array.find(array == 0)
        >>> print array_find
        [1, 4]
        >>> print array[array_find].shape
        (1, 2)

        >>> array = CArray([[1,0,-6,2,0]], tosparse=True)
        >>> array_find = array.find(array == 0)
        >>> print array_find
        [1, 4]
        >>> print array[array_find].shape
        (1, 2)

        """
        if not self.is_vector_like:
            raise ValueError("array is 2D, use find_2d() instead.")

        return self._data.find(self.__class__(condition)._data)[1]

    def find_2d(self, condition):
        """Returns array elements indices depending on condition.

        Parameters
        ----------
        condition : CArray or array_like
            Array like with booleans representing desired condition.

        Returns
        -------
        out_find : list
            List of len(out_find) == 2 with indices corresponding to
            array elements where condition is True. out_find[0] holds
            the indices of rows, out_find[1] the indices of columns.

        Notes
        -----
        Using .find_2d() output for indexing original array always result in
        a ravelled array with elements which corresponding condition was True.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray([[1,0],[-6,3],[2,7]])
        >>> array_find = array.find_2d(array > 0)
        >>> print array_find
        [[0, 1, 2, 2], [0, 1, 0, 1]]
        >>> print array[array_find]
        CArray([1 3 2 7])

        >>> array = CArray([[1,0],[-6,0],[2,0]], tosparse=True)
        >>> array_find = array.find_2d(array == 0)
        >>> print array_find
        [[0, 1, 2], [1, 1, 1]]
        >>> print array[array_find].shape
        (1, 3)

        >>> array = CArray([1,0,2])
        >>> array_find = array.find_2d(array > 0)
        >>> print array_find
        [[0, 0], [0, 2]]
        >>> print array[array_find]
        CArray([1 2])

        """
        return self._data.find(self.__class__(condition)._data)

    def bincount(self):
        """Count number of occurrences of each value in array of non-negative ints.

        Only flat arrays of integer dtype are supported.

        DENSE FORMAT ONLY

        Returns
        -------
        counts : CArray
            The occurrence number for every different element of array.
            The length of output array is equal to a.max()+1.

        Examples
        --------
        >>> from secml.array import CArray

        >>> a = CArray([1, 2, 3, 1, 6])
        >>> print a.bincount()
        CArray([0 2 1 1 0 0 1])

        """
        if self.ndim > 1:
            raise ValueError("array must be 1-Dimensional")
        return self.__class__(self._data.bincount())

    def binary_search(self, value):
        """Returns the index of each input value inside the array.

        DENSE ARRAYS ONLY

        If value is not found inside the array, the index
        of the closest value will be returned.
        Array will be flattened before search.

        Parameters
        ----------
        value : scalar or array_like
            Element or array of elements to search inside
            the flattened array.

        Returns
        -------
        out_index : int or CArray
            Position of input value, or the closest one, inside
            flattened array. If `value` is an array, a CArray
            with the position of each `value` element is returned.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[0,0.1],[0.4,1.0]]).binary_search(0.3)
        2

        >>> print CArray([1,2,3,4]).binary_search(10)
        3

        >>> print CArray([1,2,3,4]).binary_search(CArray([-10,1,2.2,10]))
        CArray([0 0 1 3])

        """
        return self._instance_array(
            self._data.binary_search(self.__class__(value)._data))

    def sign(self):
        """Returns element-wise sign of the array.

        The sign function returns -1 if x < 0, 0 if x == 0, 1 if x > 0.

        Returns
        -------
        out : CArray
            Array with sign of each element.
            Same shape as original array.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[-2,0,2]]).sign()
        CArray([[-1  0  1]])

        >>> print CArray([-2,0,2], tosparse=True).sign()  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	-1
          (0, 2)	1)

        """
        return self.__class__(self._data.sign())

    def abs(self):
        """Returns array elements without sign.

        Returns
        -------
        array : CArray
            Array with the corresponding elements without sign.

        """
        return abs(self)

    def repmat(self, m, n):
        """Repeat an array M x N times.

        Parameters
        ----------
        m, n : int
            The number of times the array is repeated along
            the first and second axes.

        Returns
        -------
        rep_array : CArray
            The result of repeating array m X n times.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[1,2]],tosparse=True).repmat(2,2)  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (0, 1)	2
          (0, 2)	1
          (0, 3)	2
          (1, 0)	1
          (1, 1)	2
          (1, 2)	1
          (1, 3)	2)

        >>> print CArray([1,2]).repmat(2,2)
        CArray([[1 2 1 2]
         [1 2 1 2]])
        >>> print CArray([1,2]).repmat(1,2)
        CArray([[1 2 1 2]])
        >>> print CArray([1,2]).repmat(2,1)
        CArray([[1 2]
         [1 2]])

        """
        return self.__class__(self._data.repmat(m, n))

    def repeat(self, repeats, axis=None):
        """Repeat elements of an array.
        
        DENSE FORMAT ONLY

        Parameters
        ----------
        repeats : int, list or CArray
            The number of repetitions for each element. If this is
            an array_like object, will be broadcasted to fit the
            shape of the given axis.
        axis : int, optional
            The axis along which to repeat values. By default, array
            is flattened before use.

        Returns
        -------
        repeated_array : CArray
            Output array which has the same shape as original array,
            except along the given axis. If axis is None, a flat array
            is returned.

        Examples
        --------
        >>> from secml.array import CArray

        >>> x = CArray([[1,2],[3,4]])
        
        >>> print x.repeat(2)
        CArray([1 1 2 2 3 3 4 4])

        >>> print x.repeat(2, axis=1)  # Repeat the columns on the right
        CArray([[1 1 2 2]
         [3 3 4 4]])
        >>> print x.repeat(2, axis=0)  # Repeat the rows on the right
        CArray([[1 2]
         [1 2]
         [3 4]
         [3 4]])

        >>> print x.repeat([1, 2], axis=0)
        CArray([[1 2]
         [3 4]
         [3 4]])

        >>> x.repeat([1, 2])  # repeats size must be consistent with axis
        Traceback (most recent call last):
            ...
        ValueError: operands could not be broadcast together with shape (4,) (2,)

        >>> x = CArray([1,2,3])
        >>> print x.repeat(2, axis=0)  # Repeat the (only) row on the right
        CArray([1 1 2 2 3 3])
        >>> print x.repeat(2, axis=1)  # No columns to repeat
        Traceback (most recent call last):
          ...
        AxisError: axis 1 is out of bounds for array of dimension 1

        """
        if isinstance(repeats, (self.__class__, list)):
            repeats = CArray(repeats).todense()._data
        elif not is_int(repeats):
            raise TypeError("`repeats` must be int, list or CArray")
        return self.__class__(self._data.repeat(repeats, axis))

    def ravel(self):
        """Return a flattened array.

        For dense format a 1-dim array, containing the
        elements of the input, is returned. For sparse
        format a (1 x array.size) array will be returned.

        A copy is made only if needed.

        Returns
        -------
        flat_array : CArray
            Output of the same dtype as a, of shape (array.size,)
            for dense format or (1,array.size) for sparse format.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[1,2],[3,4]]).ravel()
        CArray([1 2 3 4])

        >>> print CArray([[1],[2],[3]], tosparse=True).ravel()  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (0, 1)	2
          (0, 2)	3)

        """
        return self.__class__(self._data.ravel())

    def flatten(self):
        """Return a flattened copy of array.

        For dense format a 1-dim array, containing the
        elements of the input, is returned. For sparse
        format a (1 x array.size) array will be returned.

        Differently from ravel, a CArray is always returned.
        Use .ravel() after call to .flatten() to get structure
        with the minimum dimensions (e.g. scalar).

        Returns
        -------
        flat_array : CArray
            Output of the same dtype as a, of shape (array.size,)
            for dense format or (1,array.size) for sparse format.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[1,2],[3,4]]).flatten()
        CArray([1 2 3 4])

        >>> print CArray([[1],[2],[3]], tosparse=True).flatten()  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (0, 1)	2
          (0, 2)	3)

        """
        return self.__class__(
            CArray(self.ravel(), tosparse=self.issparse).deepcopy())

    def all(self, axis=None, keepdims=True):
        """Test whether all array elements along a given axis evaluate to True.

        Axis selection is available for DENSE formay only.
        For sparse format, logical operation is performed
        over all the dimensions of the array

        Parameters
        ----------
        axis : int or None, optional, dense only
            Axis or axes along which logical AND between
            elements is performed. The default (axis = None)
            is to perform a logical AND over all the dimensions
            of the input array. If axis is negative, it counts
            from the last to the first axis.
        keepdims : bool, optional, dense only
            If this is set to True (default), the result will
            broadcast correctly against the original array.
            Otherwise resulting array is flattened.

        Returns
        -------
        out_all : bool or CArray
            A new boolean or array with logical AND element-wise.

        Notes
        -----
        Not a Number (NaN), positive infinity and
        negative infinity evaluate to True because
        these are not equal to zero.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[True,False],[True,True]], tosparse=True).all()
        False

        >>> print CArray([[True,False],[True,True]]).all(axis=0)
        CArray([[ True False]])

        >>> print CArray([-1,0,2,0]).all(axis=0)
        CArray([ True False  True False])
        >>> print CArray([-1,0,2,0]).all(axis=1)
        False

        >>> import numpy as np
        >>> print CArray([np.nan, np.inf, -np.inf]).all()
        True

        """
        out_all = self._data.all(axis=axis, keepdims=keepdims) \
            if self.isdense is True else self._data.all()
        return self._instance_array(out_all)

    def any(self, axis=None, keepdims=True):
        """Test whether any array elements along a given axis evaluate to True.

        Axis selection is available for DENSE format only.
        For sparse format, logical operation is performed
        over all the dimensions of the array

        Parameters
        ----------
        axis : int or None, optional, dense only
            Axis or axes along which logical AND between
            elements is performed. The default (axis = None)
            is to perform a logical OR over all the dimensions
            of the input array. If axis is negative, it counts
            from the last to the first axis.
        keepdims : bool, optional, dense only
            If this is set to True (default), the result will
            broadcast correctly against the original array.
            Otherwise resulting array is flattened.

        Returns
        -------
        out_all : bool or CArray
            A new boolean or array with logical OR element-wise.

        Notes
        -----
        Not a Number (NaN), positive infinity and
        negative infinity evaluate to True because
        these are not equal to zero.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[True,False],[True,True]], tosparse=True).any()
        True

        >>> print CArray([[True,False],[True,False]]).any(axis=0)
        CArray([[ True False]])

        >>> print CArray([-1,0,2,0]).any(axis=0)
        CArray([ True False  True False])
        >>> print CArray([-1,0,2,0]).any(axis=1)
        True

        >>> import numpy as np
        >>> print CArray([np.nan, np.inf, -np.inf]).any()
        True

        """
        out_any = self._data.any(axis=axis, keepdims=keepdims) \
            if self.isdense is True else self._data.any()
        return self._instance_array(out_any)

    def sort(self, axis=-1, kind='quicksort'):
        """Sort an array, in-place.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. The default is -1 (the last axis).
        kind : {'quicksort', 'mergesort', 'heapsort'}, optional, dense only
            Sorting algorithm to use. Default 'quicksort'.

        Notes
        -----
        Differently from numpy, we manage flat vectors as 2-Dimensional of
        shape (1, array.size). This means that when axis=0, flat array is
        returned as is (see examples).

        For large sparse arrays is actually faster to convert to dense first.

        See Also
        --------
        numpy.sort : Description of different sorting algorithms.
        .CArray.argsort : Indirect sort.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray([5,-1,0,-3])
        >>> array.sort()
        >>> print array
        CArray([-3 -1  0  5])

        >>> array = CArray([5,-1,0,-3])
        >>> array.sort(axis=0)
        >>> print array
        CArray([ 5 -1  0 -3])

        >>> array = CArray([5,-1,0,-3])
        >>> array.sort(axis=1)
        >>> print array
        CArray([-3 -1  0  5])

        >>> array = CArray([[5,-1],[7,0],[-3,-5]], tosparse=True)
        >>> array.sort()
        >>> print array # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	-1
          (0, 1)	5
          (1, 1)	7
          (2, 0)	-5
          (2, 1)	-3)

        """
        if self.isdense:
            self._data.sort(axis=axis, kind=kind)
        else:
            self._data.sort(axis=axis)

    def argsort(self, axis=None, kind='quicksort'):
        """Returns the indices that would sort an array.

        Perform an indirect sort along the given axis using
        the algorithm specified by the kind keyword. It returns
        an array of indices of the same shape as a that index
        data along the given axis in sorted order.

        Parameters
        ----------
        axis : int, None, optional
            Axis along which to sort.
            If None (default), the flattened array is used.
        kind : {'quicksort', 'mergesort', 'heapsort'}, optional, dense only
            Sorting algorithm to use. Default 'quicksort'.

        Returns
        -------
        index_array : CArray, int
            Array of indices or single integer that sort the array
            along the specified axis. In other words, array[index_array]
            yields a sorted array.

        See Also
        --------
        numpy.sort : Description of different sorting algorithms.
        .CArray.sort : In-Place sorting of array.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([0,-3,5]).argsort()
        CArray([1 0 2])

        >>> print CArray([[0,-3],[5,1]]).argsort(axis=1)  # Sorting of each row
        CArray([[1 0]
         [1 0]])

        >>> print CArray([[0,-3],[5,1]]).argsort()  # Sorting the flattened array
        CArray([1 0 3 2])

        """
        if self.isdense:
            return self._instance_array(self._data.argsort(axis=axis,
                                                           kind=kind))
        else:
            return self._instance_array(self._data.argsort(axis=axis))

    def nan_to_num(self):
        """Replace nan with zero and inf with finite numbers.

        Replace array elements if Not a Number (NaN) with zero,
        if (positive or negative) infinity with the largest
        (smallest or most negative) floating point value that
        fits in the array dtype. All finite numbers are upcast
        to the output dtype (default float64).

        Notes
        -----
        We use the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
        This means that Not a Number is not equivalent to infinity.

        Examples
        --------
        >>> from secml.array import CArray
        >>> import numpy as np
        >>> np.set_printoptions(precision=1)

        >>> array = CArray([-1,0,1,np.nan,np.inf,-np.inf])
        >>> array.nan_to_num()
        >>> print array
        CArray([ -1.000000e+000   0.000000e+000   1.000000e+000   0.000000e+000
           1.797693e+308  -1.797693e+308])

        >>> # Restoring default print precision
        >>> np.set_printoptions(precision=8)

        """
        self._data.nan_to_num()

    def max(self, axis=None, keepdims=True):
        """Return the maximum of an array or maximum along an axis.

        Parameters
        ----------
        axis : int or None, optional
            Axis along which to operate.
            If None (default), array is flattened before use.
        keepdims : bool, optional
            If this is set to True (default), the result will
            broadcast correctly against the original array.
            Otherwise resulting array is flattened.

        Returns
        -------
        array_max : scalar_like or CArray
            Maximum of array.
            If axis is None, the result is a scalar value.
            If axis is given, the result will
             broadcast correctly against the original array.


        Notes
        -----
        - For sparse arrays all elements are taken into account (both
           zeros and non-zeros).
        - NaN values are propagated, that is if at least one item is NaN,
           the corresponding max value will be NaN as well.
        - Don't use max for element-wise comparison of 2 arrays; when
           array.shape[0] is 2, maximum(a[0], a[1]) is faster than max(axis=0).

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[-1,0],[2,0]], tosparse=True).max()
        2

        >>> print CArray([[-1,0],[2,0]]).max(axis=0)
        CArray([[2 0]])
        >>> print CArray([[-1,0],[2,0]]).max(axis=1)
        CArray([[0]
         [2]])

        >>> print CArray([-1,0,2,0]).max(axis=0)
        CArray([-1  0  2  0])
        >>> print CArray([-1,0,2,0]).max(axis=1)
        2

        >>> import numpy as np
        >>> print CArray([5,np.nan]).max()
        nan

        """
        return self._instance_array(
            self._data.max(axis=axis, keepdims=keepdims))

    def min(self, axis=None, keepdims=True):
        """Return the minimum of an array or minimum along an axis.

        Parameters
        ----------
        axis : int or None, optional
            Axis along which to operate.
            If None (default), array is flattened before use.
        keepdims : bool, optional
            If this is set to True (default), the result will
            broadcast correctly against the original array.
            Otherwise resulting array is flattened.

        Returns
        -------
        array_min : scalar_like or CArray
            Minimum of array.
            If axis is None, the result is a scalar value.
            If axis is given, the result will
             broadcast correctly against the original array.

        Notes
        -----
        - For sparse arrays all elements are taken into account (both
           zeros and non-zeros).
        - NaN values are propagated, that is if at least one item is NaN,
           the corresponding max value will be NaN as well.
        - Don't use min for element-wise comparison of 2 arrays; when
           array.shape[0] is 2, minimum(a[0], a[1]) is faster than min(axis=0).

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[-1,0],[2,0]], tosparse=True).min()
        -1

        >>> print CArray([[-2,0],[-1,0]]).min(axis=0)
        CArray([[-2  0]])
        >>> print CArray([[-2,0],[-1,0]]).min(axis=1)
        CArray([[-2]
         [-1]])

        >>> print CArray([-1,0,2,0]).min(axis=0)
        CArray([-1  0  2  0])
        >>> print CArray([-1,0,2,0]).min(axis=1)
        -1

        >>> import numpy as np
        >>> print CArray([5,np.nan]).min()
        nan

        """
        return self._instance_array(
            self._data.min(axis=axis, keepdims=keepdims))

    def nanmax(self, axis=None, keepdims=True):
        """Return the maximum of an array or maximum along an axis ignoring Nans.

        When all-NaN slices are encountered a RuntimeWarning is raised
        and Nan is returned for that slice.

        DENSE ARRAYS ONLY

        Parameters
        ----------
        axis : int or None, optional
            Axis along which to operate.
            If None (default), flattened input is used.
        keepdims : bool, optional, dense only
            If this is set to True (default), the result will
            broadcast correctly against the original array.
            Otherwise resulting array is flattened.

        Returns
        -------
        array_max : scalar_like or CArray
            Maximum of array ignoring Nans.
            If axis is None, the result is a scalar value.
            If axis is given, the result is an array of
            dimension array.ndim - 1.

        Notes
        -----
        - Don't use max for element-wise comparison of 2 arrays; when
            array.shape[0] is 2, maximum(a[0], a[1]) is faster than max(axis=0).

        Examples
        --------
        >>> from secml.array import CArray

        >>> import numpy as np
        >>> print CArray([5, np.nan]).max()
        nan

        >>> print CArray([5, np.nan]).nanmax()
        5.0

        >>> print CArray([[-1, np.nan], [np.nan, 0]]).nanmax()
        0.0

        >>> print CArray([[-1, np.nan], [np.nan, 0]]).nanmax(axis=0)
        CArray([[-1.  0.]])
        >>> print CArray([[-1, np.nan], [np.nan, 0]]).nanmax(axis=1)
        CArray([[-1.]
         [ 0.]])

        """
        return self._instance_array(
            self._data.nanmax(axis=axis, keepdims=keepdims))

    def nanmin(self, axis=None, keepdims=True):
        """Return the minimum of an array or minimum along an axis ignoring Nans.

        When all-NaN slices are encountered a RuntimeWarning is raised
        and Nan is returned for that slice.

        DENSE ARRAYS ONLY

        Parameters
        ----------
        axis : int or None, optional
            Axis along which to operate.
            If None (default), flattened input is used.
        keepdims : bool, optional, dense only
            If this is set to True (default), the result will
            broadcast correctly against the original array.
            Otherwise resulting array is flattened.

        Returns
        -------
        array_min : scalar_like or CArray
            Minimum of array ignoring nans.
            If axis is None, the result is a scalar value.
            If axis is given, the result is an array of
            dimension array.ndim - 1.

        Notes
        -----
        - Don't use min for element-wise comparison of 2 arrays; when
            array.shape[0] is 2, minimum(a[0], a[1]) is faster than min(axis=0).

        Examples
        --------
        >>> from secml.array import CArray

        >>> import numpy as np
        >>> print CArray([5, np.nan]).min()
        nan

        >>> print CArray([5, np.nan]).nanmin()
        5.0

        >>> print CArray([[-1, np.nan], [np.nan, 0]]).nanmin()
        -1.0

        >>> print CArray([[-1, np.nan], [np.nan, 0]]).nanmin(axis=0)
        CArray([[-1.  0.]])
        >>> print CArray([[-1, np.nan], [np.nan, 0]]).nanmin(axis=1)
        CArray([[-1.]
         [ 0.]])

        """
        return self._instance_array(
            self._data.nanmin(axis=axis, keepdims=keepdims))

    def maximum(self, array):
        """Element-wise maximum of array elements.

        Compare two arrays and returns a new array containing
        the element-wise maximum. If one of the elements being
        compared is a NaN, then that element is returned.
        If both elements are NaNs then the first is returned.
        The latter distinction is important for complex NaNs,
        which are defined as at least one of the real or
        imaginary parts being a NaN. The net effect is that
        NaNs are propagated.

        Parameters
        ----------
        array : CArray or array_like
            The array like object holding the elements to compare
            current array with. Must have the same shape of first
            array.

        Returns
        -------
        CArray or scalar
            The element-wise maximum between the two arrays.
            Returns scalar if both arrays have size == 1.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[-1,0],[2,0]]).maximum(CArray([[2,-1],[2,-1]]))
        CArray([[2 0]
         [2 0]])

        >>> print CArray([[-1,0],[2,0]], tosparse=True).maximum(CArray([[2,-1],[2,-1]]))  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	2
          (1, 0)	2)

        >>> print CArray([-1]).maximum(CArray([2]))
        2

        """
        other_carray = self.__class__(array)
        if not self.has_compatible_shape(other_carray):
            raise ValueError("arrays to compare must have the same shape. "
                             "{:} different from {:}."
                             "".format(self.shape, other_carray.shape))
        # Arrays have compatible shape, let's compare them
        if self.issparse:
            return self._instance_array(
                self._data.maximum(other_carray.tosparse()._data))
        else:
            return self._instance_array(
                self._data.maximum(other_carray.todense()._data))

    def minimum(self, array):
        """Element-wise minimum of array elements.

        Compare two arrays and returns a new array containing
        the element-wise minimum. If one of the elements being
        compared is a NaN, then that element is returned.
        If both elements are NaNs then the first is returned.
        The latter distinction is important for complex NaNs,
        which are defined as at least one of the real or
        imaginary parts being a NaN. The net effect is that
        NaNs are propagated.

        Parameters
        ----------
        array : CArray or array_like
            The array like object holding the elements to compare
            current array with. Must have the same shape of first
            array.

        Returns
        -------
        CArray or scalar
            The element-wise minimum between the two arrays.
            Returns scalar if both arrays have size == 1.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[-1,0],[2,0]]).minimum(CArray([[2,-1],[2,-1]]))
        CArray([[-1 -1]
         [ 2 -1]])

        >>> print CArray([[-1,0],[2,0]], tosparse=True).minimum(CArray([[2,-1],[2,-1]]))  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	-1
          (0, 1)	-1
          (1, 0)	2
          (1, 1)	-1)


        >>> print CArray([-1]).minimum(CArray([2]))
        -1

        """
        other_carray = self.__class__(array)
        if not self.has_compatible_shape(other_carray):
            raise ValueError("arrays to compare must have the same shape. "
                             "{:} different from {:}."
                             "".format(self.shape, other_carray.shape))
        # Arrays have compatible shape, let's compare them
        if self.issparse:
            return self._instance_array(
                self._data.minimum(other_carray.tosparse()._data))
        else:
            return self._instance_array(
                self._data.minimum(other_carray.todense()._data))

    def round(self, decimals=0):
        """Evenly round to the given number of decimals.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to (default: 0).
            If decimals is negative, it specifies the number of
            positions to round to the left of the decimal point.

        Returns
        -------
        out_rounded : CArray
            An new array containing the rounded values. The real and
            imaginary parts of complex numbers are rounded separately.
            The result of rounding a float is a float.

        Notes
        -----
        For values exactly halfway between rounded decimal values,
        we rounds to the nearest even value. Thus 1.5 and 2.5
        round to 2.0, -0.5 and 0.5 round to 0.0, etc. Results may
        also be surprising due to the inexact representation of
        decimal fractions in the IEEE floating point standard [1]_
        and errors introduced when scaling by powers of ten.

        References
        ----------
        .. [1] "Lecture Notes on the Status of  IEEE 754", William Kahan,
               http://www.cs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF
        .. [2] "How Futile are Mindless Assessments of
               Roundoff in Floating-Point Computation?", William Kahan,
               http://www.cs.berkeley.edu/~wkahan/Mindless.pdf

        See Also
        --------
        .CArray.ceil : Return the ceiling of the input, element-wise.
        .CArray.floor : Return the floor of the input, element-wise.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([1.28,5.62]).round()
        CArray([ 1.  6.])

        >>> print CArray([1.28,5.62],tosparse=True).round(decimals=1)  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1.3
          (0, 1)	5.6)

        >>> print CArray([.5, 1.5, 2.5, 3.5, 4.5]).round() # rounds to nearest even value
        CArray([ 0.  2.  2.  4.  4.])

        >>> print CArray([1,5,6,11]).round(decimals=-1)
        CArray([ 0  0 10 10])

        """
        return self.__class__(self._data.round(decimals))

    def ceil(self):
        """Return the ceiling of the input, element-wise.

        The ceil of the scalar x is the smallest integer i,
        such that i >= x.

        Returns
        -------
        out_ceil : CArray
            The ceiling of each element in x, with float dtype.

        See Also
        --------
        .CArray.round : Evenly round to the given number of decimals.
        .CArray.floor : Return the floor of the input, element-wise.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]).ceil()
        CArray([-1. -1. -0.  1.  2.  2.  2.])

        >>> # Array with dtype == int is upcasted to float before ceiling
        >>> print CArray([[-2, -1], [1, 1]], tosparse=True).ceil()  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	-2.0
          (0, 1)	-1.0
          (1, 0)	1.0
          (1, 1)	1.0)

        """
        return self.__class__(self._data.ceil())

    def floor(self):
        """Return the floor of the input, element-wise.

        The floor of the scalar x is the largest integer i,
        such that i <= x.

        Returns
        -------
        out_floor : CArray
            The floor of each element in x, with float dtype.

        Notes
        -----
        Some spreadsheet programs calculate the "floor-towards-zero",
        in other words floor(-2.5) == -2. We instead uses the
        definition of floor where floor(-2.5) == -3.

        See Also
        --------
        .CArray.round : Evenly round to the given number of decimals.
        .CArray.ceil : Return the ceiling of the input, element-wise.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]).floor()
        CArray([-2. -2. -1.  0.  1.  1.  2.])

        >>> # Array with dtype == int is upcasted to float before flooring
        >>> print CArray([[-2, -1], [1, 1]], tosparse=True).floor()  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	-2.0
          (0, 1)	-1.0
          (1, 0)	1.0
          (1, 1)	1.0)

        """
        return self.__class__(self._data.floor())

    def mean(self, axis=None, keepdims=True):
        """Compute the arithmetic mean along the specified axis.

        Returns the average of the array elements. The average is
        taken over the flattened array by default, otherwise over
        the specified axis. Output is casted to dtype float.

        Parameters
        ----------
        axis : int, optional
            Axis along which the means are computed.
            The default is to compute the mean of the flattened array.
        keepdims : bool, optional
            If this is set to True (default), the result will
            broadcast correctly against the original array.

        Returns
        -------
        out_mean : float or CArray
            Returns a new dense array containing the mean for givenaxis.
            If axis=None, returns a float scalar with global average of array.

        Notes
        -----
        The arithmetic mean is the sum of the elements along
        the axis divided by the number of elements.

        Note that for floating-point input, the mean is computed
        using default float precision. Depending on the input
        data, this can cause the results to be inaccurate,
        especially for 32-bit machines (float32).

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[1,4],[4,3]], tosparse=True).mean()
        3.0

        >>> print CArray([[1,4],[4,3]], tosparse=True).mean(axis=0)  # doctest: +NORMALIZE_WHITESPACE
        CArray([[ 2.5  3.5]])

        >>> print CArray([1,4,4,3]).mean(axis=0)
        CArray([ 1.  4.  4.  3.])
        >>> print CArray([1,4,4,3]).mean(axis=1)
        3.0

        """
        return self._instance_array(self._data.mean(axis=axis,
                                                    keepdims=keepdims))

    def median(self, axis=None, keepdims=True):
        """Compute the median along the specified axis.

        Given a vector V of length N, the median of V is
        the middle value of a sorted copy of V, V_sorted - i e.,
        V_sorted[(N-1)/2], when N is odd, and the average of
        the two middle values of V_sorted when N is even.

        DENSE FORMAT ONLY

        Parameters
        ----------
        axis : int, optional
            Axis along which the means are computed.
            The default is to compute the median of the flattened array.
        keepdims : bool, optional
            If this is set to True (default), the result will
            broadcast correctly against the original array.

        Returns
        -------
        out_median : float or CArray
            Returns a new array containing the median values for given
            axis or, if axis=None, return a float scalar with global
            median of array.

        Notes
        -----
        If the input contains integers or floats smaller than float64,
        then the output data-type is np.float64. Otherwise,
        the data-type of the output is the same as that of the input.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[1,4],[4,3]]).median()
        3.5

        >>> print CArray([[1,4],[4,3]]).median(axis=0)
        CArray([[ 2.5  3.5]])

        >>> print CArray([1,4,3]).median()  # array size is odd
        3.0

        >>> print CArray([1,4,4,3]).median(axis=0)
        CArray([ 1.  4.  4.  3.])
        >>> print CArray([1,4,4,3]).median(axis=1)
        3.5

        """
        return self._instance_array(
            self._data.median(axis=axis, keepdims=keepdims))

    def std(self, axis=None, ddof=0, keepdims=True):
        """Compute the standard deviation along the specified axis.

        Returns the standard deviation, a measure of the spread
        of a distribution, of the array elements. The standard
        deviation is computed for the flattened array by default,
        otherwise over the specified axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which the standard deviation is computed.
            The default is to compute the standard deviation of
            the flattened array.
        ddof : int, optional
            Means Delta Degrees of Freedom. The divisor used in
            calculations is N - ddof, where N represents the number
            of elements. By default ddof is zero.
        keepdims : bool, optional
            If this is set to True (default), the result will
            broadcast correctly against the original array.

        Returns
        -------
        out_std : float or CArray
            Returns a new array containing the standard deviation
            values for given axis or, if axis=None, return a float
            scalar with global standard deviation of array.

        Notes
        -----
        The standard deviation is the square root of the average
        of the squared deviations from the mean,
        i.e., 'std = sqrt(mean(abs(x - x.mean())**2))'.

        The average squared deviation is normally calculated as
        'x.sum() / N', where 'N = len(x)'.  If, however, `ddof` is specified,
        the divisor 'N - ddof' is used instead. In standard statistical
        practice, 'ddof=1' provides an unbiased estimator of the variance
        of the infinite population. 'ddof=0' provides a maximum likelihood
        estimate of the variance for normally distributed variables. The
        standard deviation computed in this function is the square root of
        the estimated variance, so even with 'ddof=1', it will not be an
        unbiased estimate of the standard deviation per se.

        Note that, for complex numbers, `std` takes the
        absolute value before squaring, so that the result
        is always real and not-negative.

        For floating-point input, the mean is computed using default float
        precision. Depending on the input data, this can cause the results
        to be inaccurate, especially for 32-bit machines (float32).

        Examples
        --------
        >>> from secml.array import CArray

        >>> print round(CArray([[1,4],[4,3]],tosparse=True).std(), 2)
        1.22

        >>> print CArray([[1,4],[4,3]],tosparse=True).std(axis=0)
        CArray([[ 1.5  0.5]])

        >>> print CArray([[1,4],[4,3]]).std(axis=0, ddof=1).round(2)
        CArray([[ 2.12  0.71]])

        >>> print CArray([1,4,4,3]).std(axis=0)
        CArray([ 0.  0.  0.  0.])
        >>> print round(CArray([1,4,4,3]).std(axis=1), 2)
        1.22

        """
        return self._instance_array(self._data.std(axis=axis, ddof=ddof,
                                                   keepdims=keepdims))

    def sqrt(self):
        """Compute the positive square-root of an array, element-wise.

        If any array element is complex, a complex array is returned
        (and the square-roots of negative reals are calculated). If
        all of the array elements are real, so is the resulting array,
        with negative elements returning nan.

        Returns
        -------
        out_sqrt : CArray
            A new array with the element-wise positive square-root
            of original array.

        Notes
        -----
        sqrt has, consistent with common convention, its branch cut
        the real "interval" `[-inf, 0)`, and is continuous from above
        on it. A branch cut is a curve in the complex plane across
        which a given complex function fails to be continuous.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray(2).sqrt()
        CArray([ 1.414214])

        >>> print CArray([2,3,4]).sqrt()
        CArray([ 1.414214  1.732051  2.      ])

        >>> print CArray([[2,3],[4,5]],tosparse=True).sqrt()  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1.41421356237
          (0, 1)	1.73205080757
          (1, 0)	2.0
          (1, 1)	2.2360679775)

        >>> print CArray([-3, 0]).sqrt()
        CArray([ nan   0.])

        """
        return self.__class__(self._data.sqrt())

    def logical_and(self, array):
        """Element-wise logical AND of array elements.

        Compare two arrays and returns a new array containing
        the element-wise logical AND.

        Parameters
        ----------
        array : CArray or array_like
            The array like object holding the elements to compare
            current array with. Must have the same shape of first
            array.

        Returns
        -------
        CArray
            The element-wise logical AND between the two arrays.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[-1,0],[2,0]]).logical_and(CArray([[2,-1],[2,-1]]))
        CArray([[ True False]
         [ True False]])

        >>> print CArray([-1]).logical_and(CArray([2]))
        CArray([ True])

        >>> array = CArray([1,0,2,-1])
        >>> print (array > 0).logical_and(array < 2)
        CArray([ True False False False])

        """
        if self.issparse:
            array = self.__class__(array, tosparse=True)
        else:
            array = self.__class__(array).todense()

        return self.__class__(self._data.logical_and(array._data))

    def logical_or(self, array):
        """Element-wise logical OR of array elements.

        Compare two arrays and returns a new array containing
        the element-wise logical OR.

        Parameters
        ----------
        array : CArray or array_like
            The array like object holding the elements to compare
            current array with. Must have the same shape of first
            array.

        Returns
        -------
        out_and : CArray
            The element-wise logical OR between the two arrays.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[-1,0],[2,0]]).logical_or(CArray([[2,0],[2,-1]]))
        CArray([[ True False]
         [ True  True]])

        >>> print CArray([True]).logical_or(CArray([False]))
        CArray([ True])

        >>> array = CArray([1,0,2,-1])
        >>> print (array > 0).logical_or(array < 2)
        CArray([ True  True  True  True])

        """
        if self.issparse:
            array = self.__class__(array, tosparse=True)
        else:
            array = self.__class__(array).todense()

        return self.__class__(self._data.logical_or(array._data))

    def logical_not(self):
        """Element-wise logical NOT of array elements.

        Returns
        -------
        out_not : CArray
            The element-wise logical NOT.

        Notes
        -----
        For sparse arrays this operation is usually really expensive as
        the number of zero elements is higher than the number of non-zeros.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([[-1,0],[2,0]]).logical_not()
        CArray([[False  True]
         [False  True]])

        >>> print CArray([True]).logical_not()
        CArray([False])

        >>> array = CArray([1,0,2,-1])
        >>> print (array > 0).logical_not()
        CArray([False  True False  True])

        """
        return self.__class__(self._data.logical_not())

    def inv(self):
        """Compute the (multiplicative) inverse of a square matrix.

        Given a square matrix a, return the matrix inv satisfying
        dot(array, array_inv) = dot(array_inv, array) = eye(array.shape[0]).

        Returns
        -------
        array_inv : CArray
            (Multiplicative) inverse of the square matrix.

        Raises
        ------
        LinAlgError : dense only
            If array is not square or inversion fails.
        ValueError : sparse only
            If array is not square or inversion fails

        Notes
        -----
        If the inverse of a sparse array is expected to be non-sparse,
        it will likely be faster to convert array to dense first.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray([[1., 2.], [3., 4.]])
        >>> array_inv = array.inv()
        >>> (array.dot(array_inv).round() == CArray.eye(2)).all()
        True
        >>> (array_inv.dot(array).round() == CArray.eye(2)).all()
        True

        >>> print CArray([[1., 2.], [3., 4.]], tosparse=True).inv()  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	-2.0
          (0, 1)	1.0
          (1, 0)	1.5
          (1, 1)	-0.5)

        >>> CArray([[1.,2.,3.], [4., 5.,6.]]).inv()
        Traceback (most recent call last):
            ...
        LinAlgError: Last 2 dimensions of the array must be square

        """
        return self.__class__(self._data.inv())

    def pinv(self, rcond=1e-15):
        """Compute the (Moore-Penrose) pseudo-inverse of a matrix.

        DENSE FORMAT ONLY

        Calculate the generalized inverse of a matrix using its
        singular-value decomposition (SVD) and including all
        large singular values.

        Parameters
        ----------
        rcond : float
            Cutoff for small singular values. Singular values smaller
            (in modulus) than rcond * largest_singular_value
            (again, in modulus) are set to zero.

        Returns
        -------
        array_pinv : CArray
            The pseudo-inverse of array. Resulting array have
            shape (array.shape[1], array.shape[0]).

        Raises
        ------
        LinAlgError : dense only
            If array is not square or inversion fails.

        Notes
        -----
        The pseudo-inverse of a matrix A, denoted :math:`A^+`, is defined as:
        "the matrix that 'solves' [the least-squares problem] :math:`Ax = b`,"
        i.e., if :math:`\\bar{x}` is said solution, then :math:`A^+` is that
        matrix such that :math:'\\bar{x} = A^+b'. It can be shown that if
        :math:`Q_1 \\Sigma Q_2^T = A` is the singular value decomposition of A,
        then :math:`A^+ = Q_2 \\Sigma^+ Q_1^T`, where :math:`Q_{1,2}` are
        orthogonal matrices, :math:`\\Sigma` is a diagonal matrix consisting of
        A's so-called singular values, (followed, typically, by zeros), and
        then :math:`\\Sigma^+` is simply the diagonal matrix consisting of
        the reciprocals of A's singular values (again, followed by zeros). [1]_
            
        References
        ----------
        .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
               FL, Academic Press, Inc., 1980, pp. 139-142.
               
        Examples
        --------
        >>> from secml.array import CArray

        The following example checks that:
            array * array_pinv * array == array and array_pinv * array * array_pinv == array_pinv
        >>> array = CArray([[1,3],[0,5],[8,2]])
        >>> array_pinv = array.pinv()
        >>> (array == array.dot(array_pinv.dot(array)).round()).all()
        True
        >>> (array_pinv.round(2) == array_pinv.dot(array.dot(array_pinv)).round(2)).all()
        True

        """
        return self.__class__(self._data.pinv(rcond))

    def norm(self, order=None):
        """Entrywise vector norm.

        This function provides vector norms on vector-like arrays.

        This function is able to return one of an infinite number
        of vector norms (described below), depending on the value
        of the order parameter.

        Parameters
        ----------
        order : {int, np.inf, -np.inf}, optional
            Order of the norm (see table under Notes).

        Returns
        -------
        out : float
            Norm of the array.

        Notes
        -----
        For integer order parameter, norm is computed as
        norm = sum(abs(array)**order)**(1./order). For other norm types,
        see np.norm description.

        Negative vector norms are only supported for dense arrays.

        Differently from numpy, we consider flat vectors as 2-Dimensional
        with shape (1,array.size).

        If input 2-Dimensional array is NOT vector-like,
        ValueError will be raised.

        See Also
        --------
        numpy.norm : Full description of different norms.

        Examples
        --------
        >>> from secml.array import CArray
        >>> import numpy as np

        >>> print round(CArray([1,2,3]).norm(), 5)
        3.74166
        >>> print round(CArray([[1,2,3]]).norm(2), 5)
        3.74166

        >>> print CArray([1,2,3]).norm(1)
        6.0
        >>> print CArray([1,2,3]).tosparse().norm(1)
        6.0

        >>> print CArray([1,2,3]).norm(np.inf)
        3.0
        >>> print CArray([1,2,3]).norm(-np.inf)
        1.0

        >>> print CArray([[1,2],[2,4]]).norm()
        Traceback (most recent call last):
            ...
        ValueError: Array has shape (2, 2). Call .norm_2d() to compute matricial norm or vector norm along axis.

        """
        if self.is_vector_like is False:
            raise ValueError(
                "Array has shape {:}. Call .norm_2d() to compute "
                "matricial norm or vector norm along axis.".format(self.shape))

        # Flat array to simplify dense case
        array = self.ravel()

        # 'fro' is a matrix-norm. We can exit...
        if order == 'fro':
            raise ValueError('Invalid norm order for vectors.')

        return self._instance_array(array._data.norm(order))

    def norm_2d(self, order=None, axis=None):
        """Matrix norm or vector norm along axis.

        This function provides matrix norm or vector norm along axis
        of 2D arrays. Flat arrays will be converted to 2D before
        computing the norms.

        This function is able to return one of seven different
        matrix norms, or one of an infinite number of vector norms
        (described below), depending on the value of the order parameter.

        Parameters
        ----------
        order : {'fro', non-zero int, np.inf, -np.inf}, optional
            Order of the norm (see table under Notes).
            'fro' stands for Frobenius norm.
        axis : int or None, optional
            If axis is an integer, it specifies the axis of array along
            which to compute the vector norms.
            If axis is None then the matrix norm is returned.

        Returns
        -------
        out : float or CArray
            Norm of the array. If axis is None, float is returned. Otherwise,
            a CArray with shape and number of dimensions consistent with the
            original array and the axis parameter is returned.

        Notes
        -----
        For integer order parameter, norm is computed as
        norm = sum(abs(array)**order)**(1./order). For other norm types,
        see np.norm description.
        Negative vector norms along axis are only supported for dense arrays.

        See Also
        --------
        numpy.norm : Full description of different norms.

        Examples
        --------
        >>> from secml.array import CArray
        >>> import numpy as np

        >>> print round(CArray([1,2,3]).norm_2d(), 5)
        3.74166

        >>> print CArray([1,2,3]).norm_2d(1)  # max(sum(abs(x), axis=0))
        3.0
        >>> print CArray([[1,2,3]]).norm_2d(1)
        3.0

        >>> print CArray([1,2,3]).norm_2d(np.inf)  # max(sum(abs(x), axis=1))
        6.0
        >>> print CArray([1,2,3]).norm_2d(-np.inf)  # min(sum(abs(x), axis=1))
        6.0

        >>> print CArray([[1,2],[2,4]], tosparse=True).norm_2d()
        5.0

        >>> print CArray([[1,2],[2,4]]).norm_2d(axis=0).round(5)
        CArray([[ 2.23607  4.47214]])
        >>> print CArray([[1,2],[2,4]]).norm_2d(axis=1).round(5)
        CArray([[ 2.23607]
         [ 4.47214]])

        >>> print CArray([1,2,3]).norm_2d(2, axis=0)
        CArray([[ 1.  2.  3.]])
        >>> print CArray([1,2,3]).norm_2d(2, axis=1).round(5)
        CArray([[ 3.74166]])

        >>> print CArray([1,0,3], tosparse=True).norm_2d(axis=0)  # Norm is dense
        CArray([[ 1.  0.  3.]])
        >>> print CArray([1,0,3], tosparse=True).norm_2d(axis=1).round(5)
        CArray([[ 3.16228]])

        """
        if axis is None and order in (2, -2):
            # For consistency between sparse and dense case, we block (2, -2)
            raise NotImplementedError

        if self.issparse is True:
            out = self._instance_array(
                self.atleast_2d()._data.norm_2d(order, axis=axis))
        else:
            out = self._instance_array(
                self.atleast_2d()._data.norm(order, axis=axis))

        # Return float if axis is None, else CArray
        if axis is None:
            return self._instance_array(out)
        else:
            return self.__class__(CArray(out).atleast_2d())

    def shuffle(self):
        """Modify array in-place by shuffling its contents.

        This function only shuffles the array along the first
        index of a not vector-like, multi-dimensional array.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray([2,3,0,1])
        >>> array.shuffle()
        >>> print array  # doctest: +SKIP
        CArray([0 2 1 3])  # random result

        >>> array = CArray([[2,3],[0,1]])
        >>> array.shuffle()
        >>> print array  # doctest: +SKIP
        CArray([[0 1]
         [2 3]])

        """
        self._data.shuffle()

    def atleast_2d(self):
        """View original array with at least two dimensions.

        Returns
        -------
        out : CArray
            Array with array.ndim >= 2. Copies are avoided where possible.

        Notes
        -----
        For sparse format arrays are always 2 dimensional so this method
        return a view (if possibile) of original array without any changes.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([1,2,3]).atleast_2d()
        CArray([[1 2 3]])

        """
        return self.__class__(self._data.atleast_2d())

    def sin(self):
        """Trigonometric sine, element-wise.

        DENSE FORMAT ONLY

        The array elements are considered angles, in radians
        (:math:`2\\pi` rad equals 360 degrees).

        Returns
        -------
        array_sin : CArray
            New array with trigonometric sine element-wise.

        Notes
        -----
        The sine is one of the fundamental functions of trigonometry
        (the mathematical study of triangles). Consider a circle of
        radius 1 centered on the origin. A ray comes in from the
        :math:`+x` axis, makes an angle at the origin (measured
        counter-clockwise from that axis), and departs from the
        origin. The :math:`y` coordinate of the outgoing ray's
        intersection with the unit circle is the sine of that angle.
        It ranges from -1 for :math:`x=3\\pi/2` to +1 for :math:`\\pi/2`.
        The function has zeroes where the angle is a multiple of
        :math:`\\pi`. Sines of angles between :math:`\\pi` and :math:`2\\pi`
        are negative. The numerous properties of the sine and related
        functions are included in any standard trigonometry text.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.core.constants import pi

        >>> print (CArray([0,90,180,270,360,-90,-180,-270])*pi/180).sin().round()
        CArray([ 0.  1.  0. -1. -0. -1. -0.  1.])

        >>> print (CArray([[45,135],[225,315]])*pi/180).sin()
        CArray([[ 0.707107  0.707107]
         [-0.707107 -0.707107]])

        """
        return self.__class__(self._data.sin())

    def cos(self):
        """Trigonometric cosine, element-wise.

        DENSE FORMAT ONLY

        The array elements are considered angles, in radians
        (:math:`2\\pi` rad equals 360 degrees).

        Returns
        -------
        array_cos : CArray
            New array with trigonometric cosine element-wise.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.core.constants import pi

        >>> print (CArray([0,90,180,270,360,-90,-180,-270])*pi/180).cos().round()
        CArray([ 1.  0. -1. -0.  1.  0. -1. -0.])

        >>> print (CArray([[45,135],[225,315]])*pi/180).cos()
        CArray([[ 0.707107 -0.707107]
         [-0.707107  0.707107]])

        """
        return self.__class__(self._data.cos())

    def exp(self):
        """Calculate the exponential of all elements in the input array.

        DENSE FORMAT ONLY

        Returns
        -------
        y : CArray
            New array with element-wise exponential of current data.

        Notes
        -----
        The irrational number e is also known as Euler's number. It is
        approximately 2.718281, and is the base of the natural logarithm,
        ``ln`` (this means that, if :math:`x=\\ln y=\\log_e y`, then
        :math:`e^x = y`. For real input, ``exp(x)`` is always positive.

        For complex arguments, ``x = a + ib``, we can write
        :math:`e^x = e^a e^{ib}`. The first term, :math:`e^a`, is already
        known (it is the real argument, described above). The second term,
        :math:`e^{ib}`, is :math:`\\cos b + i \\sin b`, a function with
        magnitude 1 and a periodic phase.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([0,1,3]).exp()
        CArray([  1.         2.718282  20.085537])

        """
        return self.__class__(self._data.exp())

    def log(self):
        """Calculate the natural logarithm of all elements in the input array.

        DENSE FORMAT ONLY

        Returns
        -------
        y : CArray
            New array with element-wise natural logarithm of current data.

        Notes
        -----
        Logarithm is a multivalued function: for each `x` there is an infinite
        number of `z` such that `exp(z) = x`. The convention is to return the
        `z` whose imaginary part lies in `[-pi, pi]`.

        For real-valued input data types, `log` always returns real output. For
        each value that cannot be expressed as a real number or infinity, it
        yields ``nan`` and sets the `invalid` floating point error flag.

        For complex-valued input, `log` is a complex analytical function that
        has a branch cut `[-inf, 0]` and is continuous from above on it. `log`
        handles the floating-point negative zero as an infinitesimal negative
        number, conforming to the C99 standard.

        References
        ----------
        .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
               10th printing, 1964, pp. 67. http://www.math.sfu.ca/~cbm/aands/
        .. [2] Wikipedia, "Logarithm". http://en.wikipedia.org/wiki/Logarithm

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([0,1,3]).log()
        CArray([     -inf  0.        1.098612])

        """
        return self.__class__(self._data.log())

    def log10(self):
        """Calculate the base 10 logarithm of all elements in the input array.

        DENSE FORMAT ONLY

        Returns
        -------
        y : CArray
            New array with element-wise base 10 logarithm of current data.

        Notes
        -----
        Logarithm is a multivalued function: for each `x` there is an infinite
        number of `z` such that `10**z = x`. The convention is to return the
        `z` whose imaginary part lies in `[-pi, pi]`.

        For real-valued input data types, `log10` always returns real output.
        For each value that cannot be expressed as a real number or infinity,
        it yields ``nan`` and sets the `invalid` floating point error flag.

        For complex-valued input, `log10` is a complex analytical function that
        has a branch cut `[-inf, 0]` and is continuous from above on it.
        `log10` handles the floating-point negative zero as an infinitesimal
        negative number, conforming to the C99 standard.

        References
        ----------
        .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
               10th printing, 1964, pp. 67. http://www.math.sfu.ca/~cbm/aands/
        .. [2] Wikipedia, "Logarithm". http://en.wikipedia.org/wiki/Logarithm

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([0,1,3]).log10()
        CArray([     -inf  0.        0.477121])

        """
        return self.__class__(self._data.log10())

    def pow(self, exp):
        """Array elements raised to powers from input exponent, element-wise.

        Raise each base in the array to the positionally-corresponding
        power in exp. exp must be broadcastable to the same shape of array.
        If exp is a scalar, works like standard ``**`` operator.

        Parameters
        ----------
        exp : CArray or scalar
            Exponent of power, can be another array (DENSE ONLY)
            or a single scalar. If array, must have the same
            shape of original array.

        Returns
        -------
        pow_array : CArray
            New array with the power of current data using
            input exponents.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([1,2,3]).pow(2)
        CArray([1 4 9])

        >>> print CArray([1,2,3]).pow(CArray([2,0,3]))
        CArray([ 1  1 27])

        >>> print CArray([1,0,3], tosparse=True).pow(0)  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (0, 2)	1)

        """
        return self.__pow__(exp)

    def normpdf(self, mu=0, sigma=1):
        """Return normal distribution function value with mean
        and standard deviation given for the current array values.

        The norm pdf is given by:

        .. math::

            y = f(x|\\mu,\\sigma) = \\frac{1}{\\sigma \\sqrt{2\\pi}}
            e^{\\frac{-(x-\\mu)^2}{2\\sigma^2}}

        The standard normal distribution has :math:`\\mu=0`
        and :math:`\\sigma=1`.

        Parameters
        ----------
        mu : float, optional
            Normal distribution mean. Default = 0.
        sigma : float, optional
            Normal distribution standard deviation. Default = 1.

        Returns
        -------
        y : CArray
            Normal distribution values.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray([1,2,3]).normpdf()
        CArray([ 0.241971  0.053991  0.004432])

        >>> print CArray([1,2,3]).normpdf(2,0.5)
        CArray([ 0.107982  0.797885  0.107982])

        """
        return self.__class__(self._data.normpdf(float(mu), float(sigma)))

    def interp(self, x_data, y_data, return_left=None, return_right=None):
        """One-dimensional linear interpolation.

        DENSE FORMAT ONLY

        Returns the one-dimensional piecewise linear interpolant
        to a function with given values at discrete data-points.

        Parameters
        ----------
        x_data : array_like (floats)
            Flat array of floats with the x-coordinates
            of the data points, must be increasing.
        y_data : array_like (floats)
            Flat array of floats with the y-coordinates
            of the data points, same length as `x_data`.
        return_left : float, optional
            Value to return for x < x_data[0], default is y_data[0].
        return_right : float, optional
            Value to return for x > x_data[-1], default is y_data[-1].

        Returns
        -------
        out_interp : CArray
            The interpolated values, same shape as x.

        Notes
        -----
        The function does not check that the x-coordinate sequence
        `x_data` is increasing. If `x_data` is not increasing, the
        results are nonsense.

        Examples
        --------
        .. plot::

            >>> from secml.array import CArray
            >>> from secml.figure import CFigure
            >>> from secml.core.constants import pi

            >>> fig = CFigure(fontsize=14)
            >>> x_array = CArray.linspace(0, 2*pi, 10)
            >>> y_array = x_array.sin()
            >>> x_vals = CArray.linspace(0, 2*pi, 50)

            >>> y_interp = x_vals.interp(x_array, y_array)

            >>> fig.sp.plot(x_array, y_array, 'o')
            >>> fig.sp.plot(x_vals, y_interp, '-xr')

        """
        return self.__class__(
            self._data.interp(CArray(x_data).astype(float)._data,
                              CArray(y_data).astype(float)._data,
                              return_left, return_right))

    def apply_along_axis(self, func, axis, *args, **kwargs):
        """Apply function to 1-D slices along the given axis.

        `func` should accept 1-D arrays and return a single scalar or
        a 1-D array.

        Only 1-D and 2-D arrays are currently supported.

        Parameters
        ----------
        func : function
            Function object to apply along the given axis.
            Must return a single scalar or a 1-D array.
        axis : int
            Axis along which to apply the function.
        *args, **kwargs : optional
            Any other input value for `func`.

        Returns
        -------
        out : CArray
            1-Dimensional array of size `data.shape[0]` with the
            output of `func` for each row in data. Datatype of
            output array is always float.

        Examples
        --------
        >>> from secml.array import CArray

        >>> a = CArray([[1,2],[10,20],[100,200]])

        >>> def return_sum(x):
        ...     return x.sum()

        >>> print a.apply_along_axis(return_sum, axis=0)  # Column-wise
        CArray([ 111.  222.])

        >>> print a.apply_along_axis(return_sum, axis=1)  # Row-wise
        CArray([   3.   30.  300.])

        """
        data_2d = self.atleast_2d()
        # Preallocate output array
        if axis == 0:
            out = CArray.zeros(self.shape[1])
            for i in xrange(self.shape[1]):
                out[i] = func(data_2d[:, i], *args, **kwargs)
        elif axis == 1:
            out = CArray.zeros(self.shape[0])
            for i in xrange(self.shape[0]):
                out[i] = func(data_2d[i, :], *args, **kwargs)
        else:
            raise ValueError("`apply_along_axis` currently available "
                             "for 1-D and 2-D arrays only.")

        return out

    # -------------------------------- #
    # # # # # # CLASSMETHODS # # # # # #
    # ---------------------------------#

    @classmethod
    def empty(cls, shape, dtype=float, sparse=False):
        """Return a new array of given shape and type, without filling it.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new array, e.g., 2 or (2,3).
        dtype : str or datatype, optional
            The desired data-type for the array. Default is float.
        sparse : bool, optional
            If False (default) a dense array will be returned. Otherwise,
            a sparse array is returned.

        Returns
        -------
        out_empty : CArray
            Array of arbitrary values with the given shape, dtype, and format.

        Notes
        -----
        `.empty`, unlike `.zeros`, does not set the array values to zero, and may
        therefore be marginally faster. On the other hand, it requires the user
        to manually set all the values in the array, and should be used with caution.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray.empty(3)
        >>> print array  # doctest: +SKIP
        CArray([  0.00000000e+000   4.94944794e+173   6.93660640e-310])  # random

        >>> array = CArray.empty((2,1), dtype=int, sparse=True)
        >>> print array  # doctest: +SKIP
        CArray(  (0, 0)	391726787298048
          (1, 0)	3414398447641579603)  # random
        >>> print array.shape
        (2, 1)

        """
        return cls(CDense.empty(*((shape,) if not isinstance(shape, tuple)
                                  else shape), dtype=dtype), tosparse=sparse)

    @classmethod
    def zeros(cls, shape, dtype=float, sparse=False):
        """Return a new array of given shape and type, filled with zeros.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new array, e.g., 2 or (2,3).
        dtype : str or datatype, optional
            The desired data-type for the array. Default is float.
        sparse : bool, optional
            If False (default) a dense array will be returned. Otherwise,
            a sparse array of zeros is created. Note that sparse arrays
            with only zeros apper empty when printing.

        Returns
        -------
        out_zeros : CArray
            Array of zeros with the given shape, dtype, and format.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray.zeros(2)
        CArray([ 0.  0.])

        >>> array = CArray.zeros((2,1), dtype=int, sparse=True)
        >>> print array  # sparse arrays with only zeros appear empty...
        CArray()
        >>> print array.shape
        (2, 1)

        """
        if sparse is not True:
            return cls(CDense.zeros(shape, dtype=dtype))
        else:  # Sparse case
            shape = (1, shape) if not isinstance(shape, tuple) else shape
            shape = (1, shape[0]) if len(shape) <= 1 else shape
            # We now use a secret init for CSparse... :)
            return cls(CSparse(shape, dtype=dtype))

    @classmethod
    def ones(cls, shape, dtype=float, sparse=False):
        """Return a new array of given shape and type, filled with ones.


        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new array, e.g., 2 or (2,3).
        dtype : str or datatype, optional
            The desired data-type for the array. Default is float.
        sparse : bool, optional
            If False (default) a dense array will be returned. Otherwise,
            a sparse array of ones is created.

        Returns
        -------
        out_zeros : CArray
            Array of ones with the given shape, dtype, and format.

        Notes
        -----
        When sparse is True, array is converted to sparse format
        after ones generation under dense environment. Consequently,
        for large distributions, performance is not comparable to
        classical sparse array creation routines.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray.ones(2)
        CArray([ 1.  1.])

        >>> print CArray.ones((2,1), dtype=int, sparse=True)  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (1, 0)	1)

        """
        if isinstance(shape, tuple):
            return cls(CDense.ones(*shape, dtype=dtype), tosparse=sparse)
        else:
            return cls(CDense.ones(shape, dtype=dtype), tosparse=sparse)

    @classmethod
    def eye(cls, n_rows, n_cols=None, k=0, dtype=float, sparse=False):
        """Return a 2-D array with ones on the diagonal and zeros elsewhere.

        Parameters
        ----------
        n_rows : int
            Number of rows in the output.
        n_cols : int or None, optional
            Number of columns in the output. If None, defaults to n_rows.
        k : int, optional
            Index of the diagonal: 0 (the default) refers to the main
            diagonal, a positive value refers to an upper diagonal,
            and a negative value to a lower diagonal.
        dtype : str or dtype, optional
            Data-type of the returned array.
        sparse : bool, optional
            If False (default) a dense array will be returned. Otherwise,
            a sparse array will be created.

        Returns
        -------
        out_eye : CArray
            An array where all elements are equal to zero, except for the
            k-th diagonal, whose values are equal to one.

        Examples
        --------
        >>> from secml.array import CArray
        >>> array = CArray.eye(2)
        >>> print array
        CArray([[ 1.  0.]
         [ 0.  1.]])

        >>> array = CArray.eye(2, 3, k=1, dtype=int, sparse=True)
        >>> print array  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 1)	1
          (1, 2)	1)

        >>> print array.shape
        (2, 3)

        """
        if sparse is True:
            return cls(CSparse.eye(n_rows, n_cols, k=k, dtype=dtype))
        else:
            return cls(CDense.eye(n_rows, n_cols, k=k, dtype=dtype))

    @classmethod
    def rand(cls, shape, density=0.5, random_state=None, sparse=False):
        """Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over
        the stated interval. To sample Unif[a, b), b > a multiply
        the output of rand by (b-a) and add a:

            (b - a) * rand() + a

        Parameters
        ----------
        shape : tuple of ints
            Shape of the new array.
        density : scalar, optional, sparse only
            Density of the generated sparse array, default 0.5.
            Density equal to one means a full array, density of 0 means
            no non-zero items.
        random_state : int or None, optional
            If int, random_state is the seed used by the
            random number generator; If None, is the seed used by np.random.
        sparse : bool, optional
            If False (default) a dense array will be returned. Otherwise,
            a sparse array of zeros is created.

        Returns
        -------
        out_rand : CArray
            Array of random floats with the given shape and format.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array_dense = CArray.rand(shape=(2, 3))
        >>> print array_dense  # doctest: +SKIP
        [[ 0.68588225  0.88371576  0.3958642 ]
         [ 0.58243871  0.05104796  0.77719998]]

        >>> array_sparse = CArray.rand((2, 3), sparse=True, density=0.45)
        >>> print array_sparse  # doctest: +SKIP
        CArray(  (0, 0)	0.209653887609
          (1, 1)	0.521906773406)

        """
        np.random.seed(random_state)  # Setting the random seed
        if sparse is True:
            shape = (1, shape[0]) if len(shape) <= 1 else shape
            return cls(CSparse.rand(*shape, density=density))
        else:
            return cls(CDense.rand(*shape))

    @classmethod
    def randn(cls, shape, random_state=None):
        """Return a sample (or samples) from the "standard normal" distribution.

        DENSE FORMAT ONLY

        The samples are generated from a univariate "normal"
        (Gaussian) distribution of mean 0 and variance 1.

        Parameters
        ----------
        shape : tuple of ints
            Shape of the new array.
        random_state : int or None, optional
            If int, random_state is the seed used by the
            random number generator; If None, is the seed used by np.random.

        Returns
        ----------
        out : CArray or float
            A new array of given shape with floating-point samples
            from the standard normal distribution, or a single such
            float if no parameters were supplied.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array_dense = CArray.randn(shape=(2, 3))
        >>> print array_dense  # doctest: +SKIP
        CArray([[ 0.2848132  -0.02965108  1.41184901]
         [-1.3842878   0.2673215   0.18978747]])

        """
        np.random.seed(random_state)  # Setting the random seed
        return cls(CDense.randn(*shape))

    @classmethod
    def randint(cls, low, high=None,
                shape=None, random_state=None, sparse=False):
        """Return random integers from low (inclusive) to high (exclusive).

        Return random integers from the "discrete uniform" distribution
        in the "half-open" interval [low, high). If high is None
        (the default), then results are from [0, low).

        Parameters
        ----------
        low : int
            Lowest (signed) integer to be drawn from the distribution
            (unless high=None, in which case this parameter is the
            highest such integer).
        high : int or None, optional
            If provided, one above the largest (signed) integer to be
            drawn from the distribution (see above for behavior if
            high=None).
        shape : int, tuple of ints or None, optional
            Shape of output array. If None, a single value is returned.
        random_state : int or None, optional
            If int, random_state is the seed used by the
            random number generator; If None, is the seed used by np.random.
        sparse : bool, optional
            If False (default) a dense array will be returned. Otherwise,
            a random sparse array is created.

        Returns
        -------
        out_sample : CArray
            Size-shaped array of random integers.

        Notes
        -----
        When sparse is True, array is converted to sparse format
        after random generation under dense environment. Consequently,
        for large distributions, performance is not comparable to
        classical sparse array creation routines.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray.randint(5, shape=10)
        >>> print array  # doctest: +SKIP
        CArray([1 0 0 2 2 0 2 4 3 4])

        >>> array = CArray.randint(0, 5, 10)
        >>> print array  # doctest: +SKIP
        CArray([0 2 2 0 3 1 4 2 4 1])

        >>> array = CArray.randint(0, 5, (2, 2))
        >>> print array  # doctest: +SKIP
        CArray([[3 2]
         [0 2]])

        """
        np.random.seed(random_state)  # Setting the random seed
        return cls(CDense.randint(
            low=low, high=high, size=shape), tosparse=sparse)

    @classmethod
    def randsample(cls, a, shape=None,
                   replace=False, random_state=None, sparse=False):
        """Generates a random sample from a given array.

        Parameters
        ----------
        a : CArray or int
            If an array, a random sample is generated from its
            elements. If an int, the random sample is generated
            as if a was CArray.arange(n)
        shape : int, tuple of ints or None, optional
            Shape of output array. If None, a single value is returned.
        replace : bool, optional
            Whether the sample is with or without replacement, default False.
        random_state : int or None, optional
            If int, random_state is the seed used by the
            random number generator; If None, is the seed used by np.random.
        sparse : bool, optional
            If False (default) a dense array will be returned. Otherwise,
            a random sparse array is created.

        Returns
        -------
        out_sample : CArray
            The generated random samples.

        Notes
        -----
        When sparse is True, array is converted to sparse format
        after random generation under dense environment. Consequently,
        for large distributions, performance is not comparable to
        classical sparse array creation routines.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray.randsample(10, 4)
        >>> print array  # doctest: +SKIP
        CArray([1 8 3 5])

        >>> array = CArray.randsample(CArray([1,5,6,7,3]), 4)
        >>> print array  # doctest: +SKIP
        CArray([3 7 5 6])

        >>> CArray.randsample(3, 4)
        Traceback (most recent call last):
            ...
        ValueError: Cannot take a larger sample than population when 'replace=False'

        """
        array = a if not isinstance(a, cls) else a.ravel()._data
        np.random.seed(random_state)  # Setting the random seed
        return cls(CDense.randsample(
            array=array, size=shape, replace=replace), tosparse=sparse)

    @classmethod
    def linspace(cls, start, stop, num=50, endpoint=True, sparse=False):
        """Return evenly spaced numbers over a specified interval.

        Returns num evenly spaced float samples, calculated over the
        interval [start, stop ]. The endpoint of the interval can
        optionally be excluded.

        Parameters
        ----------
        start : scalar
            The starting value of the sequence.
        stop : scalar
            The end value of the sequence, unless endpoint is set
            to False. In that case, the sequence consists of all
            but the last of num + 1 evenly spaced samples, so that
            stop is excluded. Note that the step size changes when
            endpoint is False.
        num : int, optional
            Number of samples to generate. Default is 50.
        endpoint : bool, optional
            If True, stop is the last sample. Otherwise, it is not
            included. Default is True.
        sparse : bool, optional
            If False (default) a dense array will be returned. Otherwise,
            a sparse array is created.

        Returns
        -------
        out_samples : CArray
            There are num equally spaced samples in the closed interval
            [start, stop] or the half-open interval [start, stop) (depending
            on whether endpoint is True or False).

        Notes
        -----
        When sparse is True, array is converted to sparse format
        after random generation under dense environment. Consequently,
        for large distributions, performance is not comparable to
        classical sparse array creation routines.

        See Also
        --------
        .CArray.arange : Similar to linspace, but uses a specific step size.

        Examples
        --------
        >>> from secml.array import CArray

        >>> array = CArray.linspace(3.0, 4, num=5)
        >>> print array
        CArray([ 3.    3.25  3.5   3.75  4.  ])

        >>> array = CArray.linspace(3, 4., num=5, endpoint=False)
        >>> print array
        CArray([ 3.   3.2  3.4  3.6  3.8])

        """
        return cls(CDense.linspace(
            start, stop, num=num, endpoint=endpoint), tosparse=sparse)

    @classmethod
    def arange(cls, start, stop=None, step=1, dtype=None, sparse=False):
        """Return evenly spaced values within a given interval.

        Values are generated within the half-open interval [start, stop).
        For integer arguments the function is equivalent to the Python
        built-in range function, but returns an ndarray rather than a list.

        When using a non-integer step, such as 0.1, the results will often
        not be consistent. It is better to use linspace for these cases.

        Parameters
        ----------
        start : scalar
            Lowest scalar to be drawn from the distribution (unless
            stop=None, in which case this parameter is the highest
            such scalar). The default start value is 0.
        stop : scalar, optional
            If provided, end of the interval (see above for behavior
            if stop=None). The interval does not include this value,
            except in some cases where step is not an integer and
            floating point round-off affects the length of out.
        step : scalar, optional
            Spacing between values. For any output out, this is the distance
            between two adjacent values, out[i+1] - out[i]. The default step
            size is 1. If step is specified, start must also be given.
        dtype : str or dtype, optional
            The type of the output array. If dtype is not given, infer the
            data type from the other input arguments.
        sparse : bool, optional
            If False (default) a dense array will be returned. Otherwise,
            a sparse array is created.

        Returns
        -------
        out : Carray
            Array of evenly spaced values. For floating point arguments,
            the length of the result is ceil((stop - start)/step). Because
            of floating point overflow, this rule may result in the last
            element of out being greater than stop.

        Notes
        -----
        When sparse is True, array is converted to sparse format
        after random generation under dense environment. Consequently,
        for large distributions, performance is not comparable to
        classical sparse array creation routines.

        See Also
        --------
        .CArray.linspace : Evenly spaced numbers with handling of endpoints.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray.arange(4)
        CArray([0 1 2 3])

        >>> print CArray.arange(4.0)
        CArray([ 0.  1.  2.  3.])

        >>> print CArray.arange(4.0, dtype=int)
        CArray([0 1 2 3])

        >>> print CArray.arange(0, 4)
        CArray([0 1 2 3])

        >>> print CArray.arange(0, 4, 0.8)
        CArray([ 0.   0.8  1.6  2.4  3.2])

        """
        return cls(CDense.arange(
            start=start, stop=stop, step=step, dtype=dtype), tosparse=sparse)

    @classmethod
    def concatenate(cls, array1, array2, axis=1):
        """Concatenate a sequence of arrays along the given axis.

        The arrays must have the same shape, except in the
        dimension corresponding to axis (the second, by default).

        This function preserves input masks if available.

        Parameters
        ----------
        array1 : CArray or array_like
            First array. If array1 is not an array, a CArray will be
            created before concatenating.
        array2 : CArray or array_like
            Second array. If array2 is not an array, a CArray will be
            created before concatenating.
        axis : int or None, optional
            The axis along which the arrays will be joined. Default is 1.
            If None, both arrays are ravelled before concatenation.

        Returns
        -------
        out : CArray
            The concatenated array. If first array is sparse, return sparse.

        Notes
        -----
        Differently from numpy, we manage flat vectors as 2-Dimensional of
        shape (1, array.size). Consequently, concatenation result of 2 flat
        arrays is a flat array only when axis=1.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray.concatenate([[1,2],[3,4]], [[11],[22]])
        CArray([[ 1  2 11]
         [ 3  4 22]])
        >>> print CArray.concatenate([[1,2],[3,4]], [[11,22]], axis=0)
        CArray([[ 1  2]
         [ 3  4]
         [11 22]])

        >>> print CArray.concatenate([[1,2],[3,4]], CArray([[11],[22]], tosparse=True))
        CArray([[ 1  2 11]
         [ 3  4 22]])
        >>> array = CArray.concatenate(CArray([[1,2],[3,4]], tosparse=True), [[11],[22]])
        >>> print array  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 0)	1
          (0, 1)	2
          (0, 2)	11
          (1, 0)	3
          (1, 1)	4
          (1, 2)	22)

        >>> print CArray.concatenate([1,2], [11,22])
        CArray([ 1  2 11 22])

        >>> print CArray.concatenate([1,2], [11,22], axis=0)
        CArray([[ 1  2]
         [11 22]])
        >>> print CArray.concatenate([1,2], [11,22], axis=1)
        CArray([ 1  2 11 22])

        """
        # Return sparse only if both original arrays are sparse
        if isinstance(array1, cls) and array1.issparse:
            return cls(CSparse.concatenate(
                array1._data, cls(array2, tosparse=True)._data, axis=axis))
        else:
            return cls(CDense.concatenate(
                cls(array1)._data, cls(array2).todense()._data, axis=axis))

    @classmethod
    def comblist(cls, list_of_list, dtype=float):
        """Generate a cartesian product of list of list input.
    
        Parameters
        ----------
        list_of_list : list of list
            1-D arrays to form the cartesian product of.
        dtype : str
            Datatype of output array. Default float.
    
        Returns
        -------
        out : CArray
            2-D array of shape (M, len(arrays)) containing
            cartesian products between input arrays.
    
        Examples
        --------
        >>> print CArray.comblist([[1, 2, 3], [4, 5], [6, 7]])
        CArray([[ 1.  4.  6.]
         [ 1.  4.  7.]
         [ 1.  5.  6.]
         [ 1.  5.  7.]
         [ 2.  4.  6.]
         [ 2.  4.  7.]
         [ 2.  5.  6.]
         [ 2.  5.  7.]
         [ 3.  4.  6.]
         [ 3.  4.  7.]
         [ 3.  5.  6.]
         [ 3.  5.  7.]])

        >>> print CArray.comblist([[1, 2], [3]], dtype=int)
        CArray([[1 3]
         [2 3]])

        """
        return cls(CDense.comblist(list_of_list, dtype=dtype))

    @classmethod
    def meshgrid(cls, xi, indexing='xy'):
        """Return coordinate matrices from coordinate vectors.

        Make N-D coordinate arrays for vectorized evaluations of N-D
        scalar/vector fields over N-D grids, given one-dimensional
        coordinate arrays x1, x2,..., xn.

        Parameters
        ----------
        x1,x2, ... xn : tuple of CArray or list
            1-D arrays representing the coordinates of a grid.
        indexing : {'xy', 'ij'}, optional
            Cartesian ('xy', default) or matrix ('ij') indexing of
            output. See Examples.

        Returns
        -------
        X1, X2,..., XN : tuple of CArray
            For vectors x1, x2,..., 'xn' with lengths Ni=len(xi),
            return (N1, N2, N3,...Nn) shaped arrays if indexing='ij'
            or (N2, N1, N3,...Nn) shaped arrays if indexing='xy' with
            the elements of xi repeated to fill the matrix along the
            first dimension for x1, the second for x2 and so on.

        Examples
        --------
        >>> from secml.array import CArray

        >>> x = CArray([1,3,5])
        >>> y = CArray([2,4,6])
        >>> xv, yv = CArray.meshgrid((x, y))
        >>> print xv
        CArray([[1 3 5]
         [1 3 5]
         [1 3 5]])
        >>> print yv
        CArray([[2 2 2]
         [4 4 4]
         [6 6 6]])
        
        >>> xv, yv = CArray.meshgrid((x, y), indexing='ij')
        >>> print xv
        CArray([[1 1 1]
         [3 3 3]
         [5 5 5]])
        >>> print yv
        CArray([[2 4 6]
         [2 4 6]
         [2 4 6]])

        """
        xi = tuple(x._data for x in xi)
        return tuple(cls(elem) for elem in CDense.meshgrid(xi, indexing=indexing))

    @classmethod
    def from_iterables(cls, iterables_list):
        """Build an array by chaining elements from objects in the input list.

        Parameters
        ----------
        iterables_list : list
            List of iterables to chain. Valid objects are CArrays,
            lists, tuples, and any other iterable. N-Dimensional arrays
            are flattened before chaining.

        Returns
        -------
        chained_values : CArray
            Flat CArray with all values chained from input objects.

        Examples
        --------
        >>> from secml.array import CArray

        >>> print CArray.from_iterables([[1, 2], (3, 4), CArray([5, 6])])
        CArray([1 2 3 4 5 6])

        >>> print CArray.from_iterables([CArray([1, 2]), CArray([[3, 4], [5, 6]])])
        CArray([1 2 3 4 5 6])

        """
        import itertools
        return CArray(list(itertools.chain.from_iterable(iterables_list)))


def is_array(x):
    """Return True if input is a CArray."""
    return isinstance(x, CArray)


def is_array_buffer(x):
    """Return True if CArray data buffer (CDense or CSparse)."""
    return isinstance(x, (CDense, CSparse))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
