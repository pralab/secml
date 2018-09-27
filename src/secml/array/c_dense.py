"""
.. module:: DenseArray
   :synopsis: Class for wrapping numpy.ndarray (ndarray subclassing)

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
.. moduleauthor:: Davide Maiorca <davide.maiorca@diee.unica.it>

"""
import numpy as np
import numpy.matlib
from numpy.linalg import inv, pinv
from scipy import interp as sc_interp
import scipy.sparse as scs
import matplotlib.mlab as mlab

from copy import deepcopy

from secml.core.type_utils import is_ndarray, is_list_of_lists, \
    is_list, is_slice, is_scalar, is_int, is_bool
from secml.array.array_utils import is_vector_index


class CDense(object):
    """Dense array. Encapsulation for np.ndarray."""
    __slots__ = '_data'  # CDense has only one slot for the ndarray

    def __init__(self, data=None, dtype=None, copy=False, shape=None):
        # Not implemented operators return NotImplemented
        if data is NotImplemented:
            raise TypeError("operator not implemented")
        data = [[]] if data is None else data
        # Light casting! We need the contained ndarray
        data = self._buffer_to_builtin(data)
        obj = np.array(data, dtype=dtype, copy=copy, ndmin=1)
        # numpy created an object array, maybe input is malformed?!
        if obj.dtype.char is 'O':
            raise TypeError("Array is malformed, check input data.")
        self._data = obj
        # Reshape created array if necessary
        if shape is not None and shape != self.shape:
            self._data = self.reshape(shape)._data

    def _buffer_to_builtin(self, data):
        """Convert data buffer to built-in arrays"""
        if isinstance(data, self.__class__):  # Extract np.ndarray
            return data.tondarray()
        else:
            return data

    # ------------------------------ #
    # # # # # # PROPERTIES # # # # # #
    # -------------------------------#

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype

    def transpose(self):
        """Transpose array data"""
        if len(self.shape) == 1:  # We consider flat arrays as 1 x N
            return self.__class__(
                np.transpose(self.reshape((1, self.shape[0])).tondarray()))
        else:
            return self.__class__(np.transpose(self.tondarray()))

    @property
    def T(self):
        """Transpose array data"""
        return self.transpose()

    @property
    def nnz(self):
        """Return the number of non zero elements."""
        return (self != 0).sum().tolist()[0]

    @property
    def nnz_indices(self):
        """Return a list of list that contain index of non zero elements."""
        return self.find(self != 0)

    @property
    def nnz_data(self):
        """Return non zero elements."""
        if len(self.nnz_indices[0]) == 0:
            return self.__class__([])
        return self[self.nnz_indices]

    # --------------------------- #
    # # # # # # CASTING # # # # # #
    # ----------------------------#

    def get_data(self):
        """Return stored data as a standard array object.

        Returns
        -------
        out : np.ndarray
            Array as a ndarray object.

        See Also
        --------
        `.tondarray()` : returns a np.ndarray.

        """
        return self.tondarray()

    def tondarray(self):
        """Return a np.ndarray view of current CDense."""
        return self._data

    def tocsr(self):
        """Return current CDense as a scipy.sparse.csr_matrix."""
        return scs.csr_matrix(self.tondarray())

    def tolist(self):
        """Return current CDense as a list."""
        return self._data.tolist()

    # ---------------------------- #
    # # # # # # INDEXING # # # # # #
    # -----------------------------#

    def _check_index(self, idx):
        """Consistency checks for __getitem__ and __setitem__ functions.

        List of lists (output of `find` method) is not managed by this method.

        Flat arrays are managed as 2D arrays, so 2 indices can be used.
        1 index can be used for 2D arrays with shape[0] == 1 (vector-like).

        Parameters
        ----------
        idx : object
            - CDense boolean mask
              Number of rows should be equal or higher
              than the number array's dimensions.
            - tuple of 2 or more elements. Any of the following:
                - CDense
                - Iterable built-in types (list, slice).
                - Atomic built-in types (int, bool).
                - Numpy atomic types (np.integer, np.bool_).
            - for vector-like arrays, one element between the above ones.

        """
        if isinstance(idx, CDense):
            if idx.dtype.kind == 'b':  # boolean mask
                # Numpy requires a mask with the same dims of target array
                if self.ndim == 1 and idx.ndim == 2:
                    idx = idx.ravel().tondarray()
                elif self.ndim == 2 and idx.ndim == 1:
                    idx = idx.atleast_2d().tondarray()
                else:
                    idx = idx.tondarray()

                # Check the shape of the boolean mask
                if idx.shape != self.shape:
                    raise IndexError(
                        "boolean mask must have shape {:}".format(self.shape))

            elif self.ndim == 1:
                # Converting to ndarray
                idx = idx.tondarray()

            elif self.ndim > 1 and self.shape[0] == 1:
                # Fake 2D index. Use ndarrays to mimic Matlab-like indexing
                idx = (np.asarray([0]), idx.tondarray())
                # 2D array: matlab-like indexing
                idx = np.ix_(*idx)

            else:
                raise IndexError(
                    "vector-like indexing is only applicable to flat arrays "
                    "or arrays with shape[0] == 1.")

        elif is_int(idx) or is_bool(idx):
            if self.ndim == 1:
                idx = np.asarray([idx])
                # Check the size of any boolean array
                self._check_index_bool(idx)

            elif self.ndim > 1 and self.shape[0] == 1:
                # Fake 2D index. Use ndarrays to mimic Matlab-like indexing
                idx = (np.asarray([0]), np.asarray([idx]))
                # Check the size of any boolean array
                self._check_index_bool(idx)
                # 2D array: matlab-like indexing
                idx = np.ix_(*idx)

            else:
                raise IndexError(
                    "vector-like indexing is only applicable to flat arrays "
                    "or arrays with shape[0] == 1.")

        elif is_list(idx):
            if self.ndim == 1:
                idx = np.asarray(idx)
                # Check the size of any boolean array
                self._check_index_bool(idx)

            elif self.ndim > 1 and self.shape[0] == 1:
                # Fake 2D index. Use ndarrays to mimic Matlab-like indexing
                idx = (np.asarray([0]), np.asarray(idx))
                # Check the size of any boolean array
                self._check_index_bool(idx)
                # 2D array: matlab-like indexing
                idx = np.ix_(*idx)

            else:
                raise IndexError(
                    "vector-like indexing is only applicable to flat arrays "
                    "or arrays with shape[0] == 1.")

        elif is_slice(idx):
            if self.ndim == 1:
                # Check validity of slice
                self._check_index_slice(0, idx)

            elif self.ndim > 1 and self.shape[0] == 1:
                # Check validity of slice
                self._check_index_slice(1, idx)
                # Fake 2D index
                idx = (np.asarray([0]), idx)

            else:
                raise IndexError(
                    "vector-like indexing is only applicable to flat arrays "
                    "or arrays with shape[0] == 1.")

        elif isinstance(idx, tuple):

            # Tuple will be now transformed to be managed directly by numpy

            if self.ndim == 1:
                # We now check the first index if is suitable for flat arrays

                if isinstance(idx[0], CDense):
                    idx_0 = idx[0].tondarray()
                else:
                    idx_0 = idx[0]

                if not is_vector_index(idx_0):
                    raise IndexError(
                        "{:} is not a valid index for axis 0".format(idx_0))

                # First index is ok, work on the 2nd
                idx_list = [idx[1]]
            else:
                idx_list = [idx[0], idx[1]]  # Use list to change indices type

            for e_i, e in enumerate(idx_list):
                # Check each tuple element and convert to ndarray
                if isinstance(e, CDense):
                    idx_list[e_i] = e.tondarray()
                    # Check the size of any boolean array inside tuple
                    t = [None, None]  # Fake index for booleans check
                    t[e_i] = idx_list[e_i]
                    self._check_index_bool(tuple(t))

                elif is_list(e):
                    idx_list[e_i] = np.asarray(e)
                    # Check the size of any boolean array inside tuple
                    t = [None, None]  # Fake index for booleans check
                    t[e_i] = idx_list[e_i]
                    self._check_index_bool(tuple(t))

                elif is_int(e):
                    idx_list[e_i] = np.asarray([e])

                elif is_bool(e):
                    idx_list[e_i] = np.asarray([e])
                    # Check the size of any boolean array inside tuple
                    t = [None, None]  # Fake index for booleans check
                    t[e_i] = idx_list[e_i]
                    self._check_index_bool(tuple(t))

                elif is_slice(e):  # slice excluded
                    self._check_index_slice(e_i, e)
                    idx_list[e_i] = e

                else:
                    raise TypeError("{:} should not be used for "
                                    "CSparse indexing.".format(type(idx)))

            # Converting back to tuple
            idx = tuple(idx_list)

            # SPECIAL CASE: Matlab-like indexing (not for 1D)
            if self.ndim > 1 and all(is_ndarray(elem) for elem in idx):
                idx = np.ix_(*idx)

        else:
            # No other object is accepted for CSparse indexing
            raise TypeError("{:} should not be used for "
                            "CDense indexing.".format(type(idx)))

        return idx

    def _check_index_bool(self, idx):
        """Check boolean array size.

        Parameters
        ----------
        idx : tuple or ndarray
            Array of booleans or tuple  to check.

        Raises
        ------
        IndexError : if array has not the same size of target axis.

        """
        # Converting atomic indices to tuple
        idx = (idx, None) if not isinstance(idx, tuple) else idx

        for elem_idx, elem in enumerate(idx):
            # boolean arrays in tuple (cross-indices) must be 1-Dimensional
            if elem is not None and elem.dtype.kind == 'b' and \
                    elem.size != self.shape[elem_idx]:
                raise IndexError(
                    "boolean index array for axis {:} must have "
                    "size {:}.".format(elem_idx, self.shape[elem_idx]))

    def _check_index_slice(self, elem_idx, elem):
        """Check slice index bounds.

        Parameters
        ----------
        elem_idx : int
            Target axis for slice index.
        elem : slice
            Slice index to check.

        Raises
        ------
        IndexError : if slice operation is out of array's bounds.

        """
        if elem.start is not None:
            # The trigger is always 0 except when
            # elem.start and self.shape[elem_idx] have same module
            # but opposite sign. Slices behave differently for
            # negatives and positives
            trig = abs(elem.start) == self.shape[elem_idx] and \
                elem.start != self.shape[elem_idx]
            if abs(elem.start) > self.shape[elem_idx] + trig - 1:
                raise IndexError(
                    "start element of index {:}, slice({:}, {:}, {:}),"
                    " is out of bounds.".format(elem_idx, elem.start,
                                                elem.step, elem.stop))
        if elem.stop is not None:
            if elem.stop > self.shape[elem_idx]:
                raise IndexError(
                    "stop element of index {:}, slice({:}, {:}, {:}), "
                    "is out of bounds.".format(elem_idx, elem.start,
                                               elem.step, elem.stop))

    def __getitem__(self, idx):
        """Redefinition of the get operation."""
        if is_list_of_lists(idx):
            # Natively supported for multi-dimensional (not flat) arrays
            return self.__class__(
                np.ndarray.__getitem__(self.atleast_2d().tondarray(), idx))

        # Check index for all other cases
        idx = self._check_index(idx)

        # We are ready for numpy
        return self.__class__(np.ndarray.__getitem__(self.tondarray(), idx))

    def __setitem__(self, idx, value):
        """Redefinition of the set operation."""
        # Check for setitem value
        if isinstance(value, CDense):
            value = value.tondarray()
        elif not (is_scalar(value) or is_bool(value)):
            raise TypeError("{:} cannot be used for setting "
                            "a CDense.".format(type(value)))

        if is_list_of_lists(idx):
            # Natively supported for multi-dimensional (not flat) arrays
            np.ndarray.__setitem__(self.atleast_2d().tondarray(), idx, value)
            return

        # Check index for all other cases
        idx = self._check_index(idx)

        # We are ready for numpy
        np.ndarray.__setitem__(self.tondarray(), idx, value)

    # ------------------------------------ #
    # # # # # # SYSTEM OVERLOADS # # # # # #
    # -------------------------------------#

    def __add__(self, other):
        """Element-wise addition.

        Parameters
        ----------
        other : CDense or scalar or bool
            Element to add to current array. If a CDense, element-wise
            addition will be performed. If scalar or boolean, the element
            will be sum to each array element.

        Returns
        -------
        array : CDense
            Array after addition.

        """
        if is_scalar(other) or is_bool(other) or isinstance(other, CDense):
            return self.__class__(
                np.add(self.tondarray(), self._buffer_to_builtin(other)))
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
        array : CDense
            Array after addition.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(np.add(other, self.tondarray()))
        else:
            return NotImplemented

    def __sub__(self, other):
        """Element-wise subtraction.

        Parameters
        ----------
        other : CDense or scalar or bool
            Element to subtract to current array. If a CDense, element-wise
            subtraction will be performed. If scalar or boolean, the element
            will be subtracted to each array element.

        Returns
        -------
        array : CDense
            Array after subtraction.

        """
        if is_scalar(other) or is_bool(other) or isinstance(other, CDense):
            return self.__class__(
                np.subtract(self.tondarray(), self._buffer_to_builtin(other)))
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
        array : CDense
            Array after subtraction.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(np.subtract(other, self.tondarray()))
        else:
            return NotImplemented

    def __mul__(self, other):
        """Element-wise product.

        Parameters
        ----------
        other : CDense or scalar or bool
            Element to multiplied to current array. If a CDense, element-wise
            product will be performed. If scalar or boolean, the element
            will be multiplied to each array element.

        Returns
        -------
        array : CDense
            Array after product.

        """
        if is_scalar(other) or is_bool(other) or isinstance(other, CDense):
            return self.__class__(
                np.multiply(self.tondarray(), self._buffer_to_builtin(other)))
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Element-wise (inverse) product.

        Parameters
        ----------
        other : scalar or bool
            Element to multiplied to current array.
            The element will be multiplied to each array element.

        Returns
        -------
        array : CDense
            Array after product.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(np.multiply(other, self.tondarray()))
        else:
            return NotImplemented

    def __truediv__(self, other):
        """Element-wise true division.

        Parameters
        ----------
        other : CDense or scalar or bool
            Element to divided to current array. If a CDense, element-wise
            division will be performed. If scalar or boolean, the element
            will be divided to each array element.

        Returns
        -------
        array : CDense
            Array after division.

        """
        if is_scalar(other) or is_bool(other) or isinstance(other, CDense):
            return self.__class__(
                np.true_divide(self.tondarray(), self._buffer_to_builtin(other)))
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """Element-wise (inverse) true division.

        Parameters
        ----------
        other : scalar or bool
            Element to divided to current array.
            The element will be divided to each array element.

        Returns
        -------
        array : CDense
            Array after division.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(np.true_divide(other, self.tondarray()))
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
        """Element-wise floor division.

        Parameters
        ----------
        other : CDense or scalar or bool
            Element to divided to current array. If a CDense, element-wise
            division will be performed. If scalar or boolean, the element
            will be divided to each array element.

        Returns
        -------
        array : CDense
            Array after division.

        """
        # Result of Numpy floor division is not reliable
        # (nan in place of inf, etc.)... Let's floor the truediv result
        out_truediv = self.__truediv__(other)
        if out_truediv is NotImplemented:
            return NotImplemented
        else:  # Return the integer part of the truediv result
            return out_truediv.floor()

    def __rfloordiv__(self, other):
        """Element-wise (inverse) floor division.

        Parameters
        ----------
        other : scalar or bool
            Element to divided to current array.
            The element will be divided to each array element.

        Returns
        -------
        array : CDense
            Array after division.

        """
        # Result of Numpy floor division is not reliable
        # (nan in place of inf, etc.)... Let's floor the truediv result
        out_truediv = self.__rtruediv__(other)
        if out_truediv is NotImplemented:
            return NotImplemented
        else:  # Return the integer part of the truediv result
            return out_truediv.floor()

    def __abs__(self):
        """Returns array elements without sign.

        Returns
        -------
        array : CDense
            Array with the corresponding elements without sign.

        """
        return self.__class__(np.abs(self.tondarray()))

    def __neg__(self):
        """Returns array elements with negated sign.

        Returns
        -------
        array : CDense
            Array with the corresponding elements with negated sign.

        """
        return self.__class__(np.negative(self.tondarray()))

    def __pow__(self, power):
        """Element-wise power.

        Parameters
        ----------
        power : CDense or scalar or bool
            Power to use. If scalar or boolean, each array element will be
            elevated to power. If a CDense, each array element will be
            elevated to the corresponding element of the input array.

        Returns
        -------
        array : CDense
            Array after power.

        """
        if is_scalar(power) or is_bool(power) or isinstance(power, CDense):
            return self.__class__(
                self.tondarray().__pow__(self._buffer_to_builtin(power)))
        else:
            return NotImplemented

    def __rpow__(self, power):
        """Element-wise (inverse) power.

        Parameters
        ----------
        power : scalar or bool
            Power to use. Each array element will be elevated to power.

        Returns
        -------
        array : CDense
            Array after power.

        """
        if is_scalar(power) or is_bool(power):
            return self.__class__(self.tondarray().__rpow__(power))
        else:
            return NotImplemented

    def __eq__(self, other):
        """Element-wise == operator.

        Parameters
        ----------
        other : CDense or scalar or bool
            Element to be compared. If a CDense, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CDense
            Boolean array with comparison result.

        """
        if is_scalar(other) or is_bool(other) or isinstance(other, CDense):
            return self.__class__(
                self.tondarray() == self._buffer_to_builtin(other))
        else:
            return NotImplemented

    def __lt__(self, other):
        """Element-wise < operator.

        Parameters
        ----------
        other : CDense or scalar or bool
            Element to be compared. If a CDense, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CDense
            Boolean array with comparison result.

        """
        if is_scalar(other) or is_bool(other) or isinstance(other, CDense):
            return self.__class__(
                self.tondarray() < self._buffer_to_builtin(other))
        else:
            return NotImplemented

    def __le__(self, other):
        """Element-wise <= operator.

        Parameters
        ----------
        other : CDense or scalar or bool
            Element to be compared. If a CDense, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CDense
            Boolean array with comparison result.

        """
        if is_scalar(other) or is_bool(other) or isinstance(other, CDense):
            return self.__class__(
                self.tondarray() <= self._buffer_to_builtin(other))
        else:
            return NotImplemented

    def __gt__(self, other):
        """Element-wise > operator.

        Parameters
        ----------
        other : CDense or scalar or bool
            Element to be compared. If a CDense, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CDense
            Boolean array with comparison result.

        """
        if is_scalar(other) or is_bool(other) or isinstance(other, CDense):
            return self.__class__(
                self.tondarray() > self._buffer_to_builtin(other))
        else:
            return NotImplemented

    def __ge__(self, other):
        """Element-wise >= operator.

        Parameters
        ----------
        other : CDense or scalar or bool
            Element to be compared. If a CDense, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CDense
            Boolean array with comparison result.

        """
        if is_scalar(other) or is_bool(other) or isinstance(other, CDense):
            return self.__class__(
                self.tondarray() >= self._buffer_to_builtin(other))
        else:
            return NotImplemented

    def __ne__(self, other):
        """Element-wise != operator.

        Parameters
        ----------
        other : CDense or scalar or bool
            Element to be compared. If a CDense, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CDense
            Boolean array with comparison result.

        """
        if is_scalar(other) or is_bool(other) or isinstance(other, CDense):
            return self.__class__(
                self.tondarray() != self._buffer_to_builtin(other))
        else:
            return NotImplemented

    def __bool__(self):
        """Manage 'and' and 'or' operators."""
        return bool(self._data)

    __nonzero__ = __bool__  # Compatibility with python < 3

    def __str__(self):
        return str(self._data).replace('array', 'CDense', 1)

    def __repr__(self):
        return repr(self._data).replace('array', 'CDense', 1)

    def __iter__(self):
        """Yields array elements in raster-scan order."""
        # The following can be simplified by ravelling the array first
        # But as .ravel() can return a copy, we prefer this
        n_rows = 1 if self.ndim == 1 else self.shape[0]
        n_columns = self.size if self.ndim == 1 else self.shape[1]
        for row_id in xrange(n_rows):
            for column_id in xrange(n_columns):
                yield self[row_id, column_id]

    # ------------------------------ #
    # # # # # # COPY UTILS # # # # # #
    # -------------------------------#

    def __copy__(self):
        """As numpy does, we return a deepcopy instead of a shallow copy."""
        return self.__class__(deepcopy(self._data))

    def __deepcopy__(self, memo):
        return self.__class__(deepcopy(self._data, memo))

    # ----------------------------- #
    # # # # # # SAVE/LOAD # # # # # #
    # ------------------------------#

    def save(self, datafile, overwrite=False):
        """Save array data into plain text file.

        Data is stored preserving original data type.

        Parameters
        ----------
        datafile : str, file_handle
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
        Flat arrays are stored with shape N x 1.

        """
        if not isinstance(self.dtype, np.dtype):
            fmt = '%s'
        elif issubclass(self.dtype.type, str):
            fmt = '%s'
        elif issubclass(self.dtype.type, int):
            fmt = '%d'
        else:  # Everything else will be stored as float
            fmt = '%f'

        # We now check if input file already exists
        import os
        if isinstance(datafile, str) and os.path.isfile(
                datafile) is True and overwrite is False:
            raise IOError("File {:} already exists. Specify overwrite=True "
                          "or delete the file.".format(datafile))

        try:
            np.savetxt(
                datafile, self.atleast_2d().tondarray(), delimiter=' ', fmt=fmt)
        except IOError as e:  # Prevent stopping after standard IOError
            print e

    @classmethod
    def load(cls, datafile, dtype=float, startrow=0, skipend=0, cols=None):
        """Load array data from plain text file.

        Parameters
        ----------
        datafile : str, file_handle
            File or filename to read. If the filename extension
            is gz or bz2, the file is first decompressed.
        dtype : str, dtype, optional
            Data type of the resulting array, default 'float'. If None,
            the dtype will be determined by the contents of the file.
        startrow : int, optional
            Array row to start loading from.
        skipend : int, optional
            Number of lines to skip from the end of the file when reading.
        cols : {Cndarray, int, tuple, slice}, optional
            Columns to load from target file.

        Returns
        -------
        loaded : CDense
            Array resulting from loading, 2-dimensional.

        """
        # Indexing for array columns to load (tuple)
        if isinstance(cols, cls):
            cols = tuple(cols.astype(int).tolist())
        try:
            return cls(np.atleast_2d(np.genfromtxt(datafile,
                                                   dtype=dtype,
                                                   delimiter=' ',
                                                   skip_header=startrow,
                                                   skip_footer=skipend,
                                                   usecols=cols)))
        except IOError as e:  # Handling standard IOError
            raise IOError(e)
        except (IndexError, StopIteration):  # Something wrong with indexing
            raise IndexError("check startrow or cols parameters")

    # ----------------------------- #
    # # # # # # UTILITIES # # # # # #
    # ------------------------------#

    def ravel(self, order=None):
        """Wrapper for numpy ravel"""
        return self.__class__(np.ravel(self.tondarray(), order))

    def reshape(self, shape):
        """Reshape array."""
        return self.__class__(self.tondarray().reshape(shape))

    def resize(self, new_shape, constant=0):
        """Return a new array with the specified shape."""
        if is_scalar(new_shape):  # Compatibility between N and (N, )
            new_shape = (new_shape,)
        # Compute size of of output array
        out_size = new_shape[0] if \
            len(new_shape) == 1 else new_shape[0] * new_shape[1]
        if out_size > self.size:  # Append missing elements
            old_array = self.ravel().append(
                self.__class__.ones(out_size - self.size,
                                    dtype=self.dtype) * constant)
        else:  # Caching old array
            old_array = self

        a_resize = np.resize(old_array.tondarray(), new_shape=new_shape)
        return self.__class__(a_resize, dtype=self.dtype)

    def astype(self, newtype):
        """Return current CDense with specified type."""
        return self.__class__(self._data.astype(newtype))

    def dot(self, other):

        if len(self.shape) + len(other.shape) != 2:  # Matrix multiplication
            # Reshaping flat vectors to 1 x N (row vectors)
            array1 = self.reshape((1, self.shape[0])) if \
                len(self.shape) == 1 else self
            array2 = other.reshape((1, other.shape[0])) if \
                len(other.shape) == 1 else other

        else:  # Inner product between flat arrays
            array1 = self
            array2 = other

        return self.__class__(np.dot(array1.tondarray(), array2.tondarray()))

    def find(self, condition):
        """Indices of current array with True condition.

        Returns a list (of lists, if the array has more than one dimension),
        each containing the indexes (one per dimension) of an element that
        satisfies the given condition.

        When using a list or array of N elements to index a CDense of N
        dimensions, we get the corresponding elements (standard ndarray
        indexing).

        Examples
        --------
        >>> a = CDense([[1,2,3],[4,5,6]])
        >>> idx = a.find(a > 2)
        >>> idx
        [[0, 1, 1, 1], [2, 0, 1, 2]]
        >>> a[idx]
        CDense([3, 4, 5, 6])

        """
        # size instead of shape as we just need one condition for each element
        if condition.size != self.size:
            raise ValueError("condition size must be {:}".format(self.size))
        return map(list, np.nonzero(condition.atleast_2d().tondarray()))

    def sort(self, axis=-1, kind='quicksort', order=None):
        """sort in place"""
        self.atleast_2d()._data.sort(axis=axis, kind=kind, order=order)

    def argsort(self, axis=-1, kind='quicksort', order=None):
        # Fast argsort only available for flat arrays
        if self.ndim == 1 or kind is not 'quicksort':
            return self.__class__(sorted(
                range(self.size), key=lambda x: self.__getitem__((0, x))))
        else:
            return self.__class__(
                np.argsort(self.tondarray(), axis, kind, order))

    def append(self, val, axis=None):
        """Wrapper for append."""
        out = self.__class__(np.append(self.atleast_2d().tondarray(),
                                       val.atleast_2d().tondarray(), axis))
        return out.ravel() if axis is None or (
            self.ndim <= 1 and val.ndim <= 1 and axis == 1) else out

    def unique(self, return_index=False,
               return_inverse=False, return_counts=False):
        """Wrapper for unique."""
        out = np.unique(
            self.tondarray(), return_index, return_inverse, return_counts)
        if not any([return_index, return_inverse, return_counts]):
            return self.__class__(out)
        else:
            return tuple([self.__class__(elem) for elem in out])

    def all(self, axis=None, keepdims=False):
        """Wrapper for numpy all."""
        out = np.all(self.atleast_2d().tondarray(), axis=axis, keepdims=keepdims)
        return self.__class__(out).ravel() if \
            self.ndim <= 1 or keepdims is False else self.__class__(out)

    def any(self, axis=None, keepdims=False):
        """Wrapper for numpy any."""
        out = np.any(self.atleast_2d().tondarray(), axis=axis, keepdims=keepdims)
        return self.__class__(out).ravel() if \
            self.ndim <= 1 or keepdims is False else self.__class__(out)

    def max(self, axis=None, keepdims=False):
        """Wrapper for numpy max."""
        out = np.amax(
            self.atleast_2d().tondarray(), axis=axis, keepdims=keepdims)
        return self.__class__(out).ravel() if \
            self.ndim <= 1 or keepdims is False else self.__class__(out)

    def min(self, axis=None, keepdims=False):
        """Wrapper for numpy min."""
        out = np.amin(
            self.atleast_2d().tondarray(), axis=axis, keepdims=keepdims)
        return self.__class__(out).ravel() if \
            self.ndim <= 1 or keepdims is False else self.__class__(out)

    def nanmax(self, axis=None, keepdims=False):
        """Wrapper for numpy nanmax."""
        out = np.nanmax(
            self.atleast_2d().tondarray(), axis=axis, keepdims=keepdims)
        return self.__class__(out).ravel() if \
            self.ndim <= 1 or keepdims is False else self.__class__(out)

    def nanmin(self, axis=None, keepdims=False):
        """Wrapper for numpy nanmin."""
        out = np.nanmin(
            self.atleast_2d().tondarray(), axis=axis, keepdims=keepdims)
        return self.__class__(out).ravel() if \
            self.ndim <= 1 or keepdims is False else self.__class__(out)

    def repmat(self, m, n):
        """Wrapper for repmat
        m: the number of times that we want repeat a alog axis 0 (vertical)
        n: the number of times that we want repeat a alog axis 1 (orizontal)

        Examples
        --------
        >>> from secml.array.c_dense import CDense
        >>> a0 = CDense([1])
        >>> print(a0.repmat(2, 3))
        [[1 1 1]
         [1 1 1]]

        """
        return self.__class__(np.matlib.repmat(self.tondarray(), m, n))

    def repeat(self, repeats, axis=None):
        """Wrapper for repeat.

        Parameters
        ----------
        repeats : int or Cndarray of ints
            The number of repetitions for each element. If this is
            an array_like object, will be broadcasted to fit the
            shape of the given axis.
        axis : int, optional
            The axis along which to repeat values. By default, array
            is flattened before use.

        Returns
        -------
        repeated_array : Cndarray
            Output array which has the same shape as original array,
            except along the given axis. If axis is None, a flat array
            is returned.

        Examples
        --------
        >>> from secml.array.c_dense import CDense

        >>> x = CDense([[1,2],[3,4]])
        
        >>> x.repeat(2)
        CDense([1, 1, 2, 2, 3, 3, 4, 4])
        
        >>> x.repeat(2, axis=1)
        CDense([[1, 1, 2, 2],
               [3, 3, 4, 4]])
        >>> x.repeat(2, axis=0)
        CDense([[1, 2],
               [1, 2],
               [3, 4],
               [3, 4]])
        
        >>> x.repeat([1, 2], axis=0)
        CDense([[1, 2],
               [3, 4],
               [3, 4]])

        """
        repeats = self._buffer_to_builtin(repeats)
        return self.__class__(
            np.repeat(self.tondarray(), repeats=repeats, axis=axis))

    def maximum(self, y):
        """Element-wise maximum with respect to input CDense."""
        return self.__class__(np.maximum(self.tondarray(), y.tondarray()))

    def minimum(self, y):
        """Element-wise minimum with respect to input CDense."""
        return self.__class__(np.minimum(self.tondarray(), y.tondarray()))

    def logical_and(self, y):
        """Element-wise logical & (and) with respect to input CDense."""
        return self.__class__(np.logical_and(self.tondarray(), y.tondarray()))

    def logical_or(self, y):
        """Element-wise logical | (or) with respect to input CDense."""
        return self.__class__(np.logical_or(self.tondarray(), y.tondarray()))

    def logical_not(self):
        """Element-wise logical ! (not) of array elements."""
        return self.__class__(np.logical_not(self.tondarray()))

    def norm(self, order=None, axis=None):
        """Wrapper for numpy norm."""
        if (self.ndim < 2 or axis is not None) and order == 'fro':
            # 'fro' is a matrix norm
            raise ValueError("Invalid norm order {:}.".format(order))

        if self.size == 0:
            # Special handle as few norms raise error for empty arrays
            if self.ndim == 2 and axis is None and order not in (
                        None, 'fro', np.inf, -np.inf, 1, -1, 2, -2):
                raise ValueError("Invalid norm order {:}.".format(order))
            return self.__class__([0.0])

        out = np.linalg.norm(
            self.atleast_2d().tondarray().astype(float) if axis is not None
            else self.tondarray().astype(float), order, axis)

        # Always return a CDense of floats
        out = self.__class__(out).astype(float)

        if axis is None:
            return out
        elif self.ndim <= 1:
            # custom axis and flat vectors, return a flat vector
            return out.ravel()
        elif self.ndim == 2:
            # for matrices always return a 2D array consistent with axis
            return out.atleast_2d().T if axis == 1 else out.atleast_2d()

    def sum(self, axis=None, keepdims=False):
        """Wrapper for numpy sum"""
        if self.size == 0:
            out = self.__class__([[0.0]])
        else:
            out = np.sum(
                self.atleast_2d().tondarray(), axis=axis, keepdims=keepdims)
        return self.__class__(out).ravel() if \
            self.ndim <= 1 or keepdims is False else self.__class__(out)

    def cumsum(self, axis=None, dtype=None):
        """Wrapper for numpy cumsum"""
        out = np.cumsum(self.atleast_2d().tondarray(), axis=axis, dtype=dtype)
        return self.__class__(out).ravel() if \
            self.ndim <= 1 else self.__class__(out)

    def prod(self, axis=None, dtype=None, keepdims=False):
        """Return the product of array elements over a given axis."""
        if self.size == 0:
            out = self.__class__([[1.0]], dtype=dtype)
        else:
            out = np.prod(self.atleast_2d().tondarray(),
                          axis=axis, dtype=dtype, keepdims=keepdims)
        return self.__class__(out).ravel() if \
            self.ndim <= 1 or keepdims is False else self.__class__(out)

    def mean(self, axis=None, dtype=float, keepdims=False):
        """Wrapper for mean"""
        out = np.mean(self.atleast_2d().tondarray(),
                      axis=axis, dtype=dtype, keepdims=keepdims)
        return self.__class__(out).ravel() if \
            self.ndim <= 1 or keepdims is False else self.__class__(out)

    def median(self, axis=None, keepdims=False):
        """Wrapper for median"""
        out = np.median(
            self.atleast_2d().tondarray(), axis=axis, keepdims=keepdims)
        return self.__class__(out).ravel() if \
            self.ndim <= 1 or keepdims is False else self.__class__(out)

    def std(self, axis=None, dtype=float, ddof=0, keepdims=False):
        """Wrapper for mean"""
        out = np.std(self.atleast_2d().tondarray(),
                     axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        return self.__class__(out).ravel() if \
            self.ndim <= 1 or keepdims is False else self.__class__(out)

    def sqrt(self):
        """Wrapper for np.sqrt"""
        return self.__class__(np.sqrt(self.tondarray()))

    def round(self, decimals=0):
        """Return a copy of your array rounded"""
        return self.__class__(np.around(self.tondarray(), decimals=decimals))

    def rint(self):
        """Round elements of the array to the nearest integer."""
        return self.round(decimals=0)

    def ceil(self):
        """Return the ceiling of the input, element-wise."""
        return self.__class__(np.ceil(self.tondarray()))

    def floor(self):
        """Return the floor of the input, element-wise."""
        return self.__class__(np.floor(self.tondarray()))

    def shuffle(self):
        """Wrapper for numpy.random.shuffle. In-place operation."""
        if self.size > 0:
            if self.ndim == 2 and self.shape[0] == 1:
                array = self.ravel()._data
                np.random.shuffle(array)
                self._data = array
                self._data = self.atleast_2d()._data
            else:
                np.random.shuffle(self._data)

    def diag(self, k=0):
        """Extract a diagonal or construct a diagonal array."""
        if self.ndim > 1 and (k > 0 and k > self.shape[1] - 1) or \
                (k < 0 and abs(k) > self.shape[0] - 1):
            raise ValueError("k exceeds matrix dimensions")
        return self.__class__(np.diag(self.tondarray(), k=k))

    def inv(self):
        """Compute the (multiplicative) inverse of a square matrix."""
        return self.__class__(inv(self.tondarray()))

    def pinv(self, rcond=1e-15):
        """Compute the (Moore-Penrose) pseudo-inverse of a matrix."""
        return self.__class__(pinv(self.tondarray(), rcond))

    def sign(self):
        """Return array sign element-wise"""
        return self.__class__(np.sign(self.tondarray()))

    def bincount(self):
        """Count number of occurrences of each non-negative int."""
        return self.__class__(np.bincount(self.tondarray()))

    def binary_search(self, value):
        """Returns the index of each input value inside the array.

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
        out_index : int or CDense
            Position of input value, or the closest one, inside
            flattened array. If `value` is an array, a CDense
            with the position of each `value` element is returned.

        Examples
        --------
        >>> from secml.array.c_dense import CDense

        >>> print CDense([[0,0.1],[0.4,1.0]]).binary_search(0.3)
        2

        >>> print CDense([1,2,3,4]).binary_search(10)
        3

        >>> print CDense([1,2,3,4]).binary_search(CDense([-10,1,2.2,10]))
        [0 0 1 3]

        """
        from bisect import bisect_left

        def bs_single(array, e):
            """Binary search of input scalar 'e' inside `array`."""
            pos = bisect_left(array.tolist(), e, 0, array.size)
            if pos == 0:  # workaround of zero-based python indexing
                return 0
            elif pos == array.size or \
                    abs(e - array[pos - 1]) < abs(e - array[pos]):
                return pos - 1
            else:
                return pos

        # As bisect_left returns a single index, so we should ravel the array
        out = map(lambda x: bs_single(self.ravel(), x), CDense(value))
        return CDense(out) if len(out) > 1 else out[0]

    def atleast_2d(self):
        """Force array to have at least 2 dimensions."""
        # All other not-empty arrays
        return self.__class__(np.atleast_2d(self.tondarray()))

    def nan_to_num(self):
        """Replace nan with zero and inf with finite numbers."""
        self[:, :] = self.__class__(np.nan_to_num(self.tondarray()))

    def argmin(self, axis=None):
        """Wrapper for numpy argmin"""
        out_min = self.__class__(np.argmin(
            self.tondarray() if axis is None else
            self.atleast_2d().tondarray(), axis=axis))

        return out_min if axis is None or self.ndim <= 1 else \
            (out_min.atleast_2d() if axis == 0 else out_min.atleast_2d().T)

    def argmax(self, axis=None):
        """Wrapper for numpy argmax"""
        out_max = self.__class__(np.argmax(
            self.tondarray() if axis is None else
            self.atleast_2d().tondarray(), axis=axis))

        return out_max if axis is None or self.ndim <= 1 else \
            (out_max.atleast_2d() if axis == 0 else out_max.atleast_2d().T)

    def nanargmin(self, axis=None):
        """Wrapper for numpy nanargmin"""
        out_min = self.__class__(
            np.nanargmin(self.tondarray() if axis is None else
                         self.atleast_2d().tondarray(), axis=axis))

        return out_min if axis is None or self.ndim <= 1 else \
            (out_min.atleast_2d() if axis == 0 else out_min.atleast_2d().T)

    def nanargmax(self, axis=None):
        """Wrapper for numpy nanargmax"""
        out_max = self.__class__(
            np.nanargmax(self.tondarray() if axis is None else
                         self.atleast_2d().tondarray(), axis=axis))

        return out_max if axis is None or self.ndim <= 1 else \
            (out_max.atleast_2d() if axis == 0 else out_max.atleast_2d().T)

    def clip(self, a_min, a_max):
        """Wrapper for numpy clip."""
        return self.__class__(np.clip(self.tondarray(), a_min=a_min, a_max=a_max))

    def sin(self):
        """Trigonometric sine, element-wise.

        The array elements are considered angles, in radians
        (:math:`2\\pi` rad equals 360 degrees).

        Returns
        -------
        array_sin : CDense
            New array with trigonometric sine element-wise.

        """
        return self.__class__(np.sin(self.tondarray()))

    def cos(self):
        """Trigonometric cosine, element-wise.

        The array elements are considered angles, in radians
        (:math:`2\\pi` rad equals 360 degrees).

        Returns
        -------
        array_cos : CDense
            New array with trigonometric cosine element-wise.

        """
        return self.__class__(np.cos(self.tondarray()))

    def exp(self):
        """Calculate the exponential of all elements in the input array.

        Returns
        -------
        y : CDense
            New array with element-wise exponential of current data.

        """
        return self.__class__(np.exp(self.tondarray()))

    def log(self):
        """Calculate the natural logarithm of all elements in the input array.

        Returns
        -------
        y : CDense
            New array with element-wise natural logarithm of current data.

        """
        return self.__class__(np.log(self.tondarray()))

    def log10(self):
        """Calculate the base 10 logarithm of all elements in the input array.

        Returns
        -------
        y : CDense
            New array with element-wise base 10 logarithm of current data.

        """
        return self.__class__(np.log10(self.tondarray()))

    def pow(self, exp):
        """Array elements raised to powers from input exponent, element-wise.

        Raise each base in the array to the positionally-corresponding
        power in exp. exp must be broadcastable to the same shape of array.
        If exp is a scalar, works like standard ``**`` operator.

        Parameters
        ----------
        exp : CDense or scalar
            Exponent of power, can be another array or a
            single scalar. If array, must have the same
            shape of original array.

        Returns
        -------
        pow_array : CDense
            New array with the power of current data using
            input exponents.

        """
        return self.__class__(
            np.power(self.tondarray(), self._buffer_to_builtin(exp)))

    def normpdf(self, mu=0.0, sigma=1.0):
        """Return normal distribution function value with mean
        and standard deviation given for the current array values.

        Parameters
        ----------  
        mu : float
            Normal distribution mean.
        sigma : float
            Normal distribution standard deviation.

        Returns
        -------
        y : CDense
            Normal distribution values

        """
        return self.__class__(
            mlab.normpdf(self.tondarray(), float(mu), float(sigma)))

    def interp(self, x_data, y_data, return_left=None, return_right=None):
        """One-dimensional linear interpolation.

        Returns the one-dimensional piecewise linear interpolant
        to a function with given values at discrete data-points.

        Parameters
        ----------
        x_data : Cndarray (floats)
            Flat array of floats with the x-coordinates
            of the data points, must be increasing.
        y_data : Cndarray (floats)
            Flat array of floats with the ycoordinates
            of the data points, same lenght as `x_data`.
        return_left : float, optional
            Value to return for x < x_data[0], default is y_data[0].
        return_right : float, optional
            Value to return for x > x_data[-1], default is y_data[-1].

        Returns
        -------
        out_interp : CDense
            The interpolated values, same shape as x.

        Notes
        -----
        The function does not check that the x-coordinate sequence
        `x_data` is increasing. If `x_data` is not increasing, the
        results are nonsense.

        """
        return self.__class__(sc_interp(self.tondarray(),
                                        x_data.ravel().tondarray(),
                                        y_data.ravel().tondarray(),
                                        return_left, return_right))

    # -------------------------------- #
    # # # # # # CLASSMETHODS # # # # # #
    # ---------------------------------#

    @classmethod
    def zeros(cls, shape, **dtype):
        """Return an array of desired shape with zeros.
        See numpy.zeros for more informations.

        Parameters
        ----------
        shape : shape of array, integer or sequence of integers.
        dtype : datatype of array data.

        Returns
        -------
        Array of zeros with desired shape.

        Examples
        --------
        >>> from secml.array.c_dense import CDense
        >>> array = CDense.zeros(2)
        >>> print array
        [ 0.  0.]
        >>> print array.shape
        (2,)

        >>> array = CDense.zeros((2, 1), dtype=int)
        >>> print array
        [[0]
         [0]]
        >>> print array.shape
        (2, 1)

        """
        return cls(np.zeros(shape, **dtype))

    @classmethod
    def ones(cls, *shape, **dtype):
        """Return an array of desired shape with ones.
        See numpy.ones for more informations.

        Parameters
        ----------
        shape : shape of array, integer or sequence of integers.
        dtype : datatype of array data.

        Returns
        -------
        Array of ones with desired shape.

        Examples
        --------
        >>> from secml.array.c_dense import CDense
        >>> array = CDense.ones(2)
        >>> print array
        [ 1.  1.]
        >>> print array.shape
        (2,)

        >>> array = CDense.ones(2, 1, dtype=int)
        >>> print array
        [[1]
         [1]]
        >>> print array.shape
        (2, 1)

        """
        return cls(np.ones(shape, **dtype))

    @classmethod
    def empty(cls, *shape, **dtype):
        """Return an (theoretically) empty array of desired shape.
        See numpy.empty for more informations.

        Parameters
        ----------
        shape : shape of array, integer or sequence of integers.
        dtype : datatype of array data.

        Returns
        -------
        Empty array with desired shape.

        Examples
        --------
        >>> from secml.array.c_dense import CDense
        >>> array = CDense.empty(2)
        >>> print array  # doctest: +SKIP
        [  6.94292784e-310   6.94292784e-310]
        >>> print array.shape
        (2,)

        >>> array = CDense.empty(2, 1, dtype=int)
        >>> print array  # doctest: +SKIP
        [[              0]
         [140526427175696]]
        >>> print array.shape
        (2, 1)

        """
        return cls(np.empty(shape, **dtype))

    @classmethod
    def eye(cls, n_rows, n_cols=None, k=0, dtype=float):
        """Return an array of desired dimension with ones on the diagonal and zeros elsewhere.
        See numpy.eye for more informations.

        Parameters
        ----------
        n_rows : number of rows for output array, integer.
        n_cols : number of columns in the output. If None, defaults to n_rows.
        k : index of the diagonal. 0 (the default) refers to the main diagonal,
            a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
        dtype : datatype of array data.

        Returns
        -------
        Array of desired shape with ones on the diagonal and zeros elsewhere.

        Examples
        --------
        >>> from secml.array.c_dense import CDense
        >>> array = CDense.eye(2)
        >>> print array
        [[ 1.  0.]
         [ 0.  1.]]
        >>> print array.shape
        (2, 2)

        >>> array = CDense.eye(2, k=1, dtype=int)
        >>> print array
        [[0 1]
         [0 0]]
        >>> print array.shape
        (2, 2)

        """
        return cls(np.eye(n_rows, n_cols, k=k, dtype=dtype))

    @classmethod
    def linspace(cls, start, stop, num=50, endpoint=True):
        """parameter: (start, stop, num, endpoint)"""
        return cls(np.linspace(start, stop, num=num, endpoint=endpoint))

    @classmethod
    def rand(cls, *shape):
        """Wrapper for random.random_sample.

        Creates a random array with random values in [0, 1)
        with desired shape (sequence of integers).

        Examples
        --------
        >>> from secml.array.c_dense import CDense
        >>> array = CDense.rand(2, 3)
        >>> print(array)  # doctest: +SKIP
        [[ 0.68588225  0.88371576  0.3958642 ]
         [ 0.58243871  0.05104796  0.77719998]]

        """
        return cls(np.random.random_sample(shape))

    @classmethod
    def randn(cls, *shape):
        """Return a sample (or samples) from the "standard normal" distribution.

        The samples are generated from a univariate "normal"
        (Gaussian) distribution of mean 0 and variance 1.

        Parameters
        ----------
        shape : int, optional
            The dimensions of the returned array, should be all
            positive. If no argument is given a single Python float
            is returned.

        Returns
        -------
        Z : CndArray or float
            A new array of given shape with floating-point samples
            from the standard normal distribution, or a single such
            float if no parameters were supplied.

        """
        return cls(np.random.randn(*shape))

    @classmethod
    def randint(cls, low, high=None, size=None):
        """Wrapper for random.randint.

        Creates a random array with random integers in [low, high) interval.
        High value (if specified) is excluded.size : int or tuple of ints, optional

        size: Output shape. If the given shape is, e.g., (m, n, k),
        then m * n * k samples are drawn. Default is None, in which case a single value is returned.

        Examples
        --------
        >>> from secml.array.c_dense import CDense
        >>> array = CDense.randint(0, 5, 10)
        >>> print(array)  # doctest: +SKIP
        [1 0 2 0 4 3 0 2 4 2]

        """
        return cls(np.random.randint(low, high, size))

    @classmethod
    def randsample(cls, array, size=None, replace=False):
        """Wrapper for random.choice.

        Generates a random sample from a given 1-D array
        High value (if specified) is excluded.size : int or tuple of ints, optional

        size: Output shape. If the given shape is, e.g., (m, n, k),
        then m * n * k samples are drawn. Default is None, in which case a single value is returned.

        Examples
        --------
        >>> from secml.array.c_dense import CDense
        >>> array = CDense.randsample(10, 4)
        >>> print(array)  # doctest: +SKIP
        [1 0 2 3]

        >>> array = CDense.randsample(CDense([1,5,6,7,3]), 4)
        >>> print(array)  # doctest: +SKIP
        [1 6 3 7]

        """
        if isinstance(array, cls):  # Cast input CDense to ndarray
            array = array.tondarray()
        return cls(np.random.choice(array, size, replace))

    @classmethod
    def concatenate(cls, array1, array2, axis=1):
        """Wrapper for ma.concatenate
        we use this and not concatenate because it preserve also np mask
        array1: CDense
        array2: CDense
        axis: int (default 1 )
        axis 0 put second array above
        axis 1 attach second array at right

        """
        # Flat arrays are transformed to 2-Dims before concatenating
        conc_array = cls(np.ma.concatenate(
            (array1.atleast_2d().tondarray(),
             array2.atleast_2d().tondarray()), axis=axis))
        # Return flat only if both array1/array2 are flat
        # and we are concatenating horizontally
        return conc_array.ravel() if \
            array1.ndim <= 1 and array2.ndim <= 1 and axis == 1 else conc_array

    @classmethod
    def arange(cls, start=None, stop=None, step=None, dtype=None):
        """Create a flatten array from 'start' to 'stop' using input 'step'."""
        return cls(np.arange(start=start, stop=stop, step=step, dtype=dtype))

    @classmethod
    def comblist(cls, list_of_list, dtype=float):
        """Generate a cartesian product of list of list input.

        Parameters
        ----------
        list_of_list : list of list
            to form the cartesian product of.
        dtype : str
            Datatype of output array. Default float.

        Returns
        -------
        out : Cndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.

        Examples
        --------
        >>> a = [[1, 2, 3], [4, 5], [6, 7]]

        >>> CDense.comblist(a)
        CDense([[ 1.,  4.,  6.],
               [ 1.,  4.,  7.],
               [ 1.,  5.,  6.],
               [ 1.,  5.,  7.],
               [ 2.,  4.,  6.],
               [ 2.,  4.,  7.],
               [ 2.,  5.,  6.],
               [ 2.,  5.,  7.],
               [ 3.,  4.,  6.],
               [ 3.,  4.,  7.],
               [ 3.,  5.,  6.],
               [ 3.,  5.,  7.]])

        """
        # Converting each list to array (skip empty arrays)
        arrays = [cls(x) for x in list_of_list if len(x) > 0]

        if len(arrays) == 0:  # Nothing to combine!
            return cls()

        # Computing number of expected combinations
        n = cls([x.size for x in arrays]).prod().tolist()[0]
        # If no custom target array is specified, initialize an array of zeros
        out = cls.zeros((n, len(arrays)), dtype=dtype)

        # Computing how many times first parameter list should be replicated
        m = n / arrays[0].size

        # Replicating first parameter values list
        out[:, 0] = arrays[0].repeat(m).transpose()

        if len(arrays) > 1:  # More than 1 list to combine, recursion!
            # Rebuild the list of lists and call recursively
            out[0:m, 1:] = cls.comblist([x.tolist() for x in arrays[1:]])
            for j in xrange(1, arrays[0].size):
                out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]

        return out

    @classmethod
    def meshgrid(cls, xi, indexing='xy'):
        """Return coordinate matrices from coordinate vectors.

        Make N-D coordinate arrays for vectorized evaluations of N-D
        scalar/vector fields over N-D grids, given one-dimensional
        coordinate arrays x1, x2,..., xn.

        Parameters
        ----------
        x1,x2, ... xn : tuple of CndArray or list
            1-D arrays representing the coordinates of a grid.
        indexing : {'xy', 'ij'}, optional
            Cartesian ('xy', default) or matrix ('ij') indexing of output.
            (only if numpy version is higher or equal than 1.7.0. )

        Returns
        -------
        X1, X2,..., XN : tuple of CndArray
            For vectors x1, x2,..., 'xn' with lengths Ni=len(xi),
            return (N1, N2, N3,...Nn) shaped arrays if indexing='ij'
            or (N2, N1, N3,...Nn) shaped arrays if indexing='xy' with
            the elements of xi repeated to fill the matrix along the
            first dimension for x1, the second for x2 and so on.

        Examples
        --------
        >>> from secml.array.c_dense import CDense

        >>> x = CDense( [1,3,5] )
        >>> y = CDense( [2,4,6] )
        >>> xv, yv = CDense.meshgrid((x, y))
        >>> xv
        CDense([[1, 3, 5],
               [1, 3, 5],
               [1, 3, 5]])
        >>> yv
        CDense([[2, 2, 2],
               [4, 4, 4],
               [6, 6, 6]])

        >>> xv, yv = CDense.meshgrid((x, y), indexing='ij')
        >>> xv
        CDense([[1, 1, 1],
               [3, 3, 3],
               [5, 5, 5]])
        >>> yv
        CDense([[2, 4, 6],
               [2, 4, 6],
               [2, 4, 6]])

        """
        xi = list(x.tondarray() if isinstance(x, cls) else x for x in xi)
        return tuple(cls(elem) for elem in np.meshgrid(*xi, indexing=indexing))
