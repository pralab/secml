"""
.. module:: SparseArray
   :synopsis: Class for wrapping scipy.sparse.csr_matrix sparse matrix format

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Davide Maiorca <davide.maiorca@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
import scipy.sparse as scs
from scipy.sparse.linalg import inv, norm
import numpy as np

from secml.array import Cdense
from secml.core.type_utils import is_ndarray, is_list_of_lists, \
    is_list, is_slice, is_scalar, is_intlike, is_int, is_bool


class Csparse(object):
    """Sparse array. Encapsulation for scipy.sparse.csr_matrix."""
    __slots__ = '_data'  # Csparse has only one slot for the scs.csr_matrix

    def __init__(self, data=None, dtype=None, copy=False, shape=None):
        """Sparse matrix initialization."""
        # Not implemented operators return NotImplemented
        if data is NotImplemented:
            raise TypeError("operator not implemented")
        data = self._buffer_to_builtin(data)
        # Reshaping is not supported for csr_matrix so we need few hacks
        new_shape = shape
        # We don't use shape when creating the buffer, but we reshape later
        if is_ndarray(data):  # This problem happens only with dense data
            shape = None
        self._data = scs.csr_matrix(data, shape, dtype, copy)
        # Now we reshape the array if needed (not available for scs.csr_matrix)
        if new_shape is not None and new_shape != self.shape:
            self._data = self.reshape(new_shape)._data

    def _buffer_to_builtin(self, data):
        """Convert data buffer to built-in arrays"""
        if isinstance(data, Cdense):  # Extract np.ndarray
            return data.toarray()
        elif isinstance(data, self.__class__):  # Extract scs.csr_matrix
            return data.tocsr()
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
        """Return total number of elements (counting both zeros and nz)."""
        return self.shape[0] * self.shape[1]

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def nnz(self):
        """Returns the number of non-zero elements."""
        return self._data.nnz

    @property
    def nnz_row_indices(self):
        """Return indices of the rows where nz elements are."""
        return Cdense([ptr_idx for ptr_idx in xrange(
            self._data.indptr.size - 1) for elem in xrange(
            self._data.indptr[ptr_idx], self._data.indptr[ptr_idx + 1])])

    @property
    def nnz_column_indices(self):
        """Return indices of the columns where nz elements are."""
        return Cdense(self._data.indices)

    @property
    def nnz_indices(self):
        """Return a list of list that contain index of non zero elements."""
        return [self.nnz_row_indices.tolist(),
                self.nnz_column_indices.tolist()]

    @property
    def nnz_data(self):
        """Return non zero elements."""
        if len(self.nnz_indices[0]) == 0:
            return self.__class__([])
        return self[self.nnz_indices]

    @property
    def T(self):
        return self.transpose()

    # --------------------------- #
    # # # # # # CASTING # # # # # #
    # ----------------------------#

    def toarray(self, order=None):
        """Convert csr_matrix to ndarray."""
        return self._data.toarray(order)

    def tocsr(self):
        """Convert to csr_matrix."""
        return self._data

    def todense(self, order=None):
        """Convert to ndarray."""
        return Cdense(self.toarray(order))

    def tolist(self):
        """Convert to list."""
        return self.todense().tolist()

    # ---------------------------- #
    # # # # # # INDEXING # # # # # #
    # -----------------------------#

    def _check_index(self, idx):
        """Consistency checks for __getitem__ and __setitem__ functions.

        1 index can be used for 2D arrays with shape[0] == 1 (vector-like).

        Parameters
        ----------
        idx : object
            - Cdense, Csparse boolean masks
              Number of rows should be equal to 2.
            - List of lists (output of `find_2d` method).
              Number of elements should be equal to 2.
            - tuple of 2 or more elements. Any of the following:
                - Cdense, Csparse
                - Iterable built-in types (list, slice).
                - Atomic built-in types (int, bool).
                - Numpy atomic types (np.integer, np.bool_).
            - for vector-like arrays, one element between the above ones.

        """
        if isinstance(idx, Cdense) or isinstance(idx, Csparse):

            # Boolean mask
            if idx.dtype.kind == 'b':

                # Boolean masks must be 2-Dimensional
                if idx.ndim == 1:
                    idx = idx.atleast_2d()

                # Convert indices to built-in arrays
                idx = idx.toarray() if isinstance(idx, Cdense) else idx
                idx = idx.tocsr() if isinstance(idx, Csparse) else idx

                # Check the shape of the boolean mask
                if idx.shape != self.shape:
                    raise IndexError(
                        "boolean mask must have shape {:}".format(self.shape))

                return idx

            # Check if array is vector-like
            if self.shape[0] != 1:
                raise IndexError("vector-like indexing is only applicable "
                                 "to arrays with shape[0] == 1.")

            # Fake 2D index. Use ndarrays to mimic Matlab-like indexing
            idx = (np.asarray([0]), idx.toarray())

            # SPECIAL CASE: Matlab-like indexing
            idx = np.ix_(*idx)

            # Workround for scipy indexing when 2 integer-like are passed
            if idx[1].size == 1:
                idx = tuple(elem.ravel()[0] for elem in idx)

        elif is_list_of_lists(idx):
            if len(idx) != 2:
                raise IndexError("for list of lists indexing, indices "
                                 "for each dimension must be provided.")
            # List of lists must be passed as a tuple
            return tuple(idx)

        # VECTOR-LIKE INDEXING (int, bool, list, slice)
        elif is_int(idx) or is_bool(idx):
            # Check if array is vector-like
            if self.shape[0] != 1:
                raise IndexError("vector-like indexing is only applicable "
                                 "to arrays with shape[0] == 1.")

            # Fake 2D index. Use ndarrays to mimic Matlab-like indexing
            idx = (np.asarray([0]), np.asarray([idx]))

            # Check the size of any boolean array inside tuple
            self._check_index_bool(idx)

            # Convert indices for get_single_element scipy method
            idx = np.ix_(*idx)
            idx = tuple(elem.ravel()[0] for elem in idx)

        elif is_list(idx):
            # Check if array is vector-like
            if self.shape[0] != 1:
                raise IndexError("vector-like indexing is only applicable "
                                 "to arrays with shape[0] == 1.")

            # Fake 2D index. Use ndarrays to mimic Matlab-like indexing
            idx = (np.asarray([0]), np.asarray(idx))

            # Check the size of any boolean array inside tuple
            self._check_index_bool(idx)

            # SPECIAL CASE: Matlab-like indexing
            idx = np.ix_(*idx)

            # Workround for scipy indexing when 2 integer-like are passed
            if idx[1].size == 1:
                idx = tuple(elem.ravel()[0] for elem in idx)

        elif is_slice(idx):
            # Check if array is vector-like
            if self.shape[0] != 1:
                raise IndexError("vector-like indexing is only applicable "
                                 "to arrays with shape[0] == 1.")

            # Fake 2D index. Use ndarrays to mimic Matlab-like indexing
            idx = (np.asarray([0]), idx)

        elif isinstance(idx, tuple):

            # Tuple will be now transformed to be managed directly by numpy

            idx_list = [idx[0], idx[1]]  # Use list to change indices type

            for e_i, e in enumerate(idx_list):
                # Check each tuple element and convert to ndarray
                if isinstance(e, Cdense) or isinstance(e, Csparse):
                    idx_list[e_i] = e.toarray()
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
                    idx_list[e_i] = e

                else:
                    raise TypeError("{:} should not be used for "
                                    "Csparse indexing.".format(type(idx)))

            # Converting back to tuple
            idx = tuple(idx_list)

            # SPECIAL CASE: Matlab-like indexing
            if all(is_ndarray(elem) for elem in idx):
                idx = np.ix_(*idx)

            # Workround for scipy indexing when 2 integer-like are passed
            if all(is_intlike(elem) for elem in idx):
                idx = tuple(elem.ravel()[0] for elem in idx)

        else:
            # No other object is accepted for Csparse indexing
            raise TypeError("{:} should not be used for "
                            "Csparse indexing.".format(type(idx)))

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

    def __getitem__(self, idx):
        """Redefinition of the get (brackets) operator."""
        # Check index for all other cases
        idx = self._check_index(idx)

        # Ready for numpy
        return self.__class__(self._data.__getitem__(idx))

    def __setitem__(self, idx, value):
        """Redefinition of the get (brackets) operator."""
        # Check for setitem value
        if isinstance(value, Cdense):
            value = value.toarray()
        elif isinstance(value, Csparse):
            value = value.tocsr()
        elif not (is_scalar(value) or is_bool(value)):
            raise TypeError("{:} cannot be used for setting "
                            "a Csparse.".format(type(value)))

        # Check index for all other cases
        idx = self._check_index(idx)

        # The tuple can now be managed directly by scipy
        self._data.__setitem__(idx, value)

        # Cleaning array after setting
        self.eliminate_zeros()

    # ------------------------------------ #
    # # # # # # SYSTEM OVERLOADS # # # # # #
    # -------------------------------------#

    def __add__(self, other):
        """Element-wise addition.

        Parameters
        ----------
        other : Csparse or Cdense
            Element to add to current array.

        Returns
        -------
        array : Csparse or Cdense
            If input is a Csparse, a Csparse will be returned.
            If input is a Cdense, a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            raise NotImplementedError("adding a nonzero scalar "
                                      "to a sparse array is not supported")
        elif isinstance(other, Csparse):  # Sparse + Sparse = Sparse
            return self.__class__(self._data.__add__(other.tocsr()))
        elif isinstance(other, Cdense) \
                and other.size > 1:  # Sparse + Dense = Dense
            return Cdense(self._data.__add__(other.toarray()))
        else:
            return NotImplemented

    def __radd__(self, other):
        """Element-wise (inverse) addition."""
        raise NotImplementedError(
            "adding a nonzero scalar to a sparse array is not supported")

    def __sub__(self, other):
        """Element-wise subtraction.

        Parameters
        ----------
        other : Csparse or Cdense
            Element to subtraction to current array.

        Returns
        -------
        array : Csparse or Cdense
            If input is a Csparse, a Csparse will be returned.
            If input is a Cdense, a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            raise NotImplementedError("subtracting a nonzero scalar "
                                      "to a sparse array is not supported")
        elif isinstance(other, Csparse):  # Sparse - Sparse = Sparse
            return self.__class__(self._data.__sub__(other.tocsr()))
        elif isinstance(other, Cdense) \
                and other.size > 1:  # Sparse + Dense = Dense
            return Cdense(self._data.__sub__(other.toarray()))
        else:
            return NotImplemented

    def __rsub__(self, other):
        """Element-wise (inverse) subtraction."""
        raise NotImplementedError(
            "subtracting a nonzero scalar to a sparse array is not supported")

    def __mul__(self, other):
        """Element-wise product.

        Parameters
        ----------
        other : Csparse or Cdense or scalar or bool
            Element to multiply to current array. If an array, element-wise
            product will be performed. If scalar or boolean, the element
            will be multiplied to each array element.

        Returns
        -------
        array : Csparse
            Array after product.

        """
        if is_scalar(other) or is_bool(other) or \
                isinstance(other, (Csparse, Cdense)):  # Always Sparse
            return self.__class__(
                self._data.multiply(self._buffer_to_builtin(other)))
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
        array : Csparse
            Array after product.

        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """Element-wise true division.

        Parameters
        ----------
        other : Csparse or Cdense or scalar or bool
            Element to divide to current array. If an array, element-wise
            division will be performed. If scalar or boolean, the element
            will be divided to each array element.

        Returns
        -------
        array : Csparse or Cdense
            Array after division. Csparse if other is a scalar or bool,
            Cdense otherwise.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__div__(other))
        elif isinstance(other, (Csparse, Cdense)):
            this = self
            other = other.atleast_2d()
            if this.shape != other.shape:  # Broadcast not supported by scipy
                if this.shape[0] == other.shape[0]:  # Equal number of rows
                    if this.shape[1] == 1:
                        this = this.repmat(1, other.shape[1])
                    elif other.shape[1] == 1:
                        other = other.repmat(1, this.shape[1])
                    else:
                        raise ValueError("inconsistent shapes")

                elif this.shape[1] == other.shape[1]:  # Equal number of cols
                    if this.shape[0] == 1:
                        this = this.repmat(other.shape[0], 1)
                    elif other.shape[0] == 1:
                        other = other.repmat(this.shape[0], 1)
                    else:
                        raise ValueError("inconsistent shapes")

                else:
                    raise ValueError("inconsistent shapes")
            # Compatible shapes, call built-in div
            return Cdense(this._data.__div__(this._buffer_to_builtin(other)))
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """Element-wise (inverse) true division."""
        raise NotImplementedError(
            "dividing a scalar by a sparse array is not supported")

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

    def __abs__(self):
        """Returns array elements without sign.

        Returns
        -------
        array : Csparse
            Array with the corresponding elements without sign.

        """
        return self.__class__(self._data.__abs__())

    def __neg__(self):
        """Returns array elements with negated sign.

        Returns
        -------
        array : Cdense
            Array with the corresponding elements with negated sign.

        """
        return self.__class__(self._data.__neg__())

    def __pow__(self, power):
        """Element-wise power.

        Parameters
        ----------
        power : scalar or bool
            Power to use. Each array element will be elevated to power.

        Returns
        -------
        array : Csparse
            Array after power.

        """
        if is_scalar(power) or is_bool(power):
            return self.__class__((pow(self._data.data, power),
                                   self._data.indices, self._data.indptr),
                                  shape=self.shape)
        else:
            return NotImplemented

    def __rpow__(self, power):
        """Element-wise (inverse) power."""
        raise NotImplementedError(
            "using a sparse array as a power is not supported")

    def __eq__(self, other):
        """Element-wise == operator.

        Parameters
        ----------
        other : Csparse or Cdense or scalar or bool
            Element to be compared. If an array, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : Csparse or Cdense
            If input is a Csparse or a scalar or a bool,
            a Csparse will be returned. If input is a Cdense,
            a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__eq__(other))
        elif isinstance(other, Csparse):  # Sparse == Sparse = Sparse
            return self.__class__(self._data.__eq__(other.tocsr()))
        elif isinstance(other, Cdense):  # Sparse == Dense = Dense
            return Cdense(self._data.__eq__(other.toarray()))
        else:
            return NotImplemented

    def __lt__(self, other):
        """Element-wise < operator.

        Parameters
        ----------
        other : Csparse or Cdense or scalar or bool
            Element to be compared. If an array, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : Csparse or Cdense
            If input is a Csparse or a scalar or a bool,
            a Csparse will be returned. If input is a Cdense,
            a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__lt__(other))
        elif isinstance(other, Csparse):  # Sparse < Sparse = Sparse
            return self.__class__(self._data.__lt__(other.tocsr()))
        elif isinstance(other, Cdense):  # Sparse < Dense = Dense
            return Cdense(self._data.__lt__(other.toarray()))
        else:
            return NotImplemented

    def __le__(self, other):
        """Element-wise <= operator.

        Parameters
        ----------
        other : Csparse or Cdense or scalar or bool
            Element to be compared. If an array, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : Csparse or Cdense
            If input is a Csparse or a scalar or a bool,
            a Csparse will be returned. If input is a Cdense,
            a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__le__(other))
        elif isinstance(other, Csparse):  # Sparse <= Sparse = Sparse
            return self.__class__(self._data.__le__(other.tocsr()))
        elif isinstance(other, Cdense):  # Sparse <= Dense = Dense
            return Cdense(self._data.__le__(other.toarray()))
        else:
            return NotImplemented

    def __gt__(self, other):
        """Element-wise > operator.

        Parameters
        ----------
        other : Csparse or Cdense or scalar or bool
            Element to be compared. If an array, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : Csparse or Cdense
            If input is a Csparse or a scalar or a bool,
            a Csparse will be returned. If input is a Cdense,
            a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__gt__(other))
        elif isinstance(other, Csparse):  # Sparse > Sparse = Sparse
            return self.__class__(self._data.__gt__(other.tocsr()))
        elif isinstance(other, Cdense):  # Sparse > Dense = Dense
            return Cdense(self._data.__gt__(other.toarray()))
        else:
            return NotImplemented

    def __ge__(self, other):
        """Element-wise >= operator.

        Parameters
        ----------
        other : Csparse or Cdense or scalar or bool
            Element to be compared. If an array, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : Csparse or Cdense
            If input is a Csparse or a scalar or a bool,
            a Csparse will be returned. If input is a Cdense,
            a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__ge__(other))
        elif isinstance(other, Csparse):  # Sparse >= Sparse = Sparse
            return self.__class__(self._data.__ge__(other.tocsr()))
        elif isinstance(other, Cdense):  # Sparse >= Dense = Dense
            return Cdense(self._data.__ge__(other.toarray()))
        else:
            return NotImplemented

    def __ne__(self, other):
        """Element-wise != operator.

        Parameters
        ----------
        other : Csparse or Cdense or scalar or bool
            Element to be compared. If an array, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : Csparse or Cdense
            If input is a Csparse or a scalar or a bool,
            a Csparse will be returned. If input is a Cdense,
            a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__ne__(other))
        elif isinstance(other, Csparse):  # Sparse != Sparse = Sparse
            return self.__class__(self._data.__ne__(other.tocsr()))
        elif isinstance(other, Cdense):  # Sparse != Dense = Dense
            return Cdense(self._data.__ne__(other.toarray()))
        else:
            return NotImplemented

    def __bool__(self):
        """Manage 'and' and 'or' operators."""
        return bool(self._data)

    __nonzero__ = __bool__  # Compatibility with python < 3

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return repr(self._data)

    def __iter__(self):
        """Yields array elements in raster-scan order."""
        for row_id in xrange(self.shape[0]):
            for column_id in xrange(self.shape[1]):
                yield self[row_id, column_id]

    # ------------------------------ #
    # # # # # # COPY UTILS # # # # # #
    # -------------------------------#

    def __copy__(self):
        """Called when copy.copy(Csparse) is called.

        Consistently with numpy.ndarray,
        this returns a DEEP COPY of current array.

        """
        return self.__class__(self._data.copy())

    def __deepcopy__(self, memo):
        """Called when copy.deepcopy(Csparse) is called."""
        y = self.__class__(self._data.copy())
        memo[id(self)] = y
        return y

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
        Format of resulting text file can be confusing: differently from dense
        format, we only save non-zero data along with indices necessary to
        reconstruct original 2-dimensional array.

        """
        # We now check if input file already exists
        import os

        if os.path.isfile(datafile) is True and overwrite is False:
            raise IOError("File {:} already exists. Specify overwrite=True "
                          "or delete the file.".format(datafile))

        # Flatting data to store (for sparse this results in 1 x size arrays)
        data_cndarray = Cdense(
            self._data.data).reshape((1, self._data.data.shape[0]))
        # Converting explicitly to int as in 64 bit machines the
        # following arrays are stored with dtype == np.int32
        indices_cndarray = Cdense(self._data.indices).reshape(
            (1, self._data.indices.shape[0])).astype(int)
        indptr_cndarray = Cdense(self._data.indptr).reshape(
            (1, self._data.indptr.shape[0])).astype(int)

        # Error handling is managed by Cdense.save()
        # file will be closed exiting from context
        with open(datafile, mode='w+') as fhandle:
            data_cndarray.save(fhandle)
            indices_cndarray.save(fhandle)
            indptr_cndarray.save(fhandle)
            fhandle.write(str(self.shape[0]) + " " + str(self.shape[1]))

    @classmethod
    def load(cls, datafile, dtype=float):
        """Load array data from plain text file.

        Parameters
        ----------
        datafile : str, file_handle
            File or filename to read. If the filename extension
            is gz or bz2, the file is first decompressed.
        dtype : str, dtype, optional
            Data type of the resulting array, default 'float'. If None,
            the dtype will be determined by the contents of the file.

        Returns
        -------
        loaded : Csparse
            Array resulting from loading, 2-dimensional.

        """
        # Cdense.load() will manage IO errors
        imported_data = Cdense.load(
            datafile, dtype=dtype, startrow=0, skipend=3).ravel().toarray()
        # Indices are always integers
        imported_indices = Cdense.load(
            datafile, dtype=int, startrow=1, skipend=2).ravel().toarray()
        imported_indptr = Cdense.load(
            datafile, dtype=int, startrow=2, skipend=1).ravel().toarray()
        shape_ndarray = Cdense.load(
            datafile, dtype=int, startrow=3, skipend=0).ravel().toarray()

        return cls((imported_data, imported_indices, imported_indptr),
                   shape=(shape_ndarray[0], shape_ndarray[1]))

    # ----------------------------- #
    # # # # # # UTILITIES # # # # # #
    # ------------------------------#

    def transpose(self):
        return self.__class__(self._data.transpose())

    def astype(self, newtype):
        return self.__class__(self._data.astype(newtype))

    def eliminate_zeros(self):
        self._data.eliminate_zeros()

    def dot(self, x):
        # Only matrix multiplication is supported for sparse arrays
        if len(x.shape) == 1:  # We work with 2D arrays
            x = x.atleast_2d()
        return self.__class__(self._data.dot(self._buffer_to_builtin(x)))

    def all(self):
        """Return True if all array elements are boolean True."""
        return bool(self.size == self.nnz and self._data.data.all())

    def any(self):
        """Return True if any array element is boolean True."""
        return bool(self._data.data.any())

    def reshape(self, newshape):
        """Reshape the matrix using input shape (int or tuple of ints)."""
        if isinstance(newshape, (int, np.integer)):
            newshape = (1, newshape)
        elif len(newshape) < 2:
            newshape = (1, newshape[0])
        elif len(newshape) > 2:
            raise ValueError(
                "'shape' must be an integer or a sequence of one/two integers")

        # Coo sparse matrices have reshape method
        array_coo = self._data.tocoo()
        n_rows, n_cols = array_coo.shape
        size = n_rows * n_cols

        new_size = newshape[0] * newshape[1]
        if new_size != size:
            raise ValueError('total size of new array must be unchanged')

        flat_indices = n_cols * array_coo.row + array_coo.col
        new_row, new_col = divmod(flat_indices, newshape[1])

        return self.__class__(scs.coo_matrix(
            (array_coo.data, (new_row, new_col)), shape=newshape))

    def resize(self, new_shape, constant=0):
        """Return a new array with the specified shape."""
        raise NotImplementedError

    def ravel(self):
        """Reshape sparse matrix to 1 x size array."""
        return self.reshape((1, self.size))

    def inv(self):
        """Compute the (multiplicative) inverse of a square matrix."""
        return self.__class__(inv(self._data))

    def pinv(self, rcond=1e-15):
        """Compute the (Moore-Penrose) pseudo-inverse of a matrix."""
        raise NotImplementedError

    def norm(self, ord=None, axis=None):
        """Return the vector norm of store data."""
        if ord not in (None, 2):
            raise NotImplementedError("Only 2-Order `norm` is currently "
                                      "implemented for sparse arrays, "
                                      "convert to dense first.")

        if self.size == 0:
            # Special handle for empty arrays
            # TODO: CHECK IF FIXED IN SCIPY > 0.16
            return Cdense([0.0])

        return Cdense(self.pow(2).sum(axis)).sqrt()

    def norm_2d(self, ord=None):
        """Return the matrix norm of store data."""
        if self.size == 0:
            # Special handle as 1-norm raise error
            # TODO: CHECK IF FIXED IN SCIPY > 0.16
            if ord not in (None, 'fro', np.inf, -np.inf, 1, -1):
                if ord in (2, -2):
                    raise NotImplementedError
                raise ValueError("Invalid norm order.")
            return Cdense([0.0])

        return Cdense(norm(self.tocsr(), ord=ord))

    def shuffle(self):
        """Shuffle array data in-place."""
        if self.size > 0:
            if self.shape[0] > 1:  # only rows are shuffled
                shuffle_idx = Cdense.randsample(self.shape[0], self.shape[0])
                self[:, :] = self[shuffle_idx, :]
            else:
                shuffle_idx = Cdense.randsample(self.shape[1], self.shape[1])
                self[:, :] = self[0, shuffle_idx]

    def nan_to_num(self):
        """Replace nan with zero and inf with finite numbers."""
        self._data.data = np.nan_to_num(self._data.data)

    def unique(self):
        """Return unique array elements in dense format."""
        unique_items = [0] if self.nnz != self.size else []  # We have at least a zero?
        # Appending nonzero elements
        unique_items += np.unique(self._data.data).tolist()
        # Return unique elements with correct dtype
        return Cdense(unique_items).astype(self.dtype)

    def atleast_2d(self):
        """Force array to have 2 dimensions.
        Sparse arrays are always 2 dimensional, so original array is returned.
        """
        return self.__class__(self)

    def diag(self, k=0):
        """Extract a diagonal or construct a diagonal array."""
        if self.shape[0] == 1:  # We use diags as numpy's diag (no offset's')
            return self.__class__(scs.diags(
                self.toarray(), offsets=[k], format='csr', dtype=self.dtype))
        elif k == 0:
            return self.__class__(self.tocsr().diagonal())
        else:
            raise ValueError("Extracting the k-th diagonal is not supported.")

    def sum(self, axis=None, keepdims=True):
        """Sum of array elements over a given axis."""
        if self.size == 0:
            out_sum = Cdense([[0.0]])
        else:
            out_sum = Cdense(self._data.sum(axis))
        return \
            out_sum.ravel() if axis is None or keepdims is False else out_sum

    def cumsum(self, axis=None):
        """Return the cumulative sum of the array elements."""
        raise NotImplementedError

    def prod(self, axis=None, dtype=None, keepdims=False):
        """Return the product of array elements over a given axis."""
        if dtype is None:
            if self.dtype == bool:  # if array is bool, out is integer
                dtype = np.int_
            else:  # out dtype is equal to array dtype
                dtype = self.dtype
        if self.size == 0:
            out = self.__class__([[1.0]], dtype=dtype)
        else:
            if axis is None:  # Global product
                if self.size == self.nnz:
                    return self.__class__(self._data.data.prod(), dtype=dtype)
                else:  # If any element is zero, product is zero
                    return self.__class__(0.0, dtype=dtype)
            elif axis == 0:
                out = Csparse((1, self.shape[1]), dtype=dtype)
                c_bincount = self.nnz_column_indices.bincount()
                for e_idx, e in enumerate(c_bincount == self.shape[0]):
                    if bool(e) is True:
                        out[0, e_idx] = self[:, e_idx].todense().prod()
            elif axis == 1 or axis == -1:
                out = Csparse((self.shape[0], 1), dtype=dtype)
                c_bincount = self.nnz_row_indices.bincount()
                for e_idx, e in enumerate(c_bincount == self.shape[1]):
                    if bool(e) is True:
                        out[e_idx, 0] = self[e_idx, :].todense().prod()
            else:
                raise ValueError("axis {:} is not valid".format(axis))

        return out.ravel() if axis is None or keepdims is False else out

    def repeat(self, repeats, axis=None):
        """Repeat elements of an array."""
        raise NotImplementedError

    def max(self, axis=None, keepdims=True):
        """Max of array elements over a given axis."""
        out_max = self.__class__(self._data.max(axis=axis))
        return \
            out_max.ravel() if axis is None or keepdims is False else out_max

    def min(self, axis=None, keepdims=True):
        """Min of array elements over a given axis."""
        out_min = self.__class__(self._data.min(axis=axis))
        return \
            out_min.ravel() if axis is None or keepdims is False else out_min

    def nanmax(self, axis=None, keepdims=True):
        raise NotImplementedError

    def nanmin(self, axis=None, keepdims=True):
        raise NotImplementedError

    def mean(self, axis=None, keepdims=True):
        """Mean of array elements over a given axis."""
        out_mean = Cdense(self._data.mean(axis))
        return \
            out_mean.ravel() if axis is None or keepdims is False else out_mean

    def maximum(self, other):
        """Element-wise maximum."""
        return self.__class__(
            self._data.maximum(self._buffer_to_builtin(other)))

    def minimum(self, other):
        """Element-wise minimum."""
        return self.__class__(
            self._data.minimum(self._buffer_to_builtin(other)))

    def median(self, axis=None, keepdims=True):
        """Median of array elements over a given axis."""
        raise NotImplementedError

    def clip(self, min, max):
        """Clip arrays acording to limits."""
        raise NotImplementedError

    def round(self, decimals=0):
        """Evenly round to the given number of decimals."""
        data = np.round(self._data.data, decimals=decimals)
        return self.__class__(
            (data, self._data.indices, self._data.indptr), shape=self.shape)

    def rint(self):
        """Round elements of the array to the nearest integer."""
        return self.round(decimals=0)

    def ceil(self):
        """Return the ceiling of the input, element-wise."""
        return self.__class__(self._data.ceil())

    def floor(self):
        """Return the floor of the input, element-wise."""
        return self.__class__(self._data.floor())

    def sqrt(self):
        """Return the element-wise square root of array."""
        return self.__class__(self._data.sqrt())

    def sign(self):
        """Return the element-wise sign of array."""
        return self.__class__(self._data.sign())

    def std(self, axis=None, ddof=0, keepdims=True):
        """Standard deviation of matrix over the given axis."""
        array_mean = self.mean(axis=axis).atleast_2d()

        centered_array = self - array_mean.repmat(
            [1 if array_mean.shape[0] == self.shape[0] else self.shape[0]][0],
            [1 if array_mean.shape[1] == self.shape[1] else self.shape[1]][0])
        # n is array size for axis == None or
        # the number of rows/columns of specified axis
        n = self.size if axis is None else self.shape[axis]
        variance = (1.0 / (float(n) - ddof)) * (centered_array ** 2)

        return Cdense(variance.sum(axis=axis, keepdims=keepdims).sqrt())

    def pow(self, exp):
        """Array elements raised to powers from input exponent, element-wise.

        Equivalent to standard ``**`` operator.

        Parameters
        ----------
        exp : scalar
            Exponent of power, single scalar.

        Returns
        -------
        pow_array : Csparse
            New array with the power of current data using
            input exponents.

        """
        return self.__pow__(exp)

    def append(self, array2, axis=None):
        """Append an  arrays along the given axis."""
        # If axis is None we simulate numpy flattening
        if axis is None:
            return self.__class__.concatenate(self.ravel(),
                                              array2.ravel(), axis=1)
        else:
            return self.__class__.concatenate(self, array2, axis)

    def repmat(self, m, n):
        """Wrapper for repmat
        m: the number of times that we want repeat a alog axis 0 (vertical)
        n: the number of times that we want repeat a alog axis 1 (orizontal)
        """
        self_csr = self.tocsr()
        rows = [self_csr for _ in xrange(n)]
        blocks = [rows for _ in xrange(m)]
        if len(blocks) == 0:  # To manage the m = 0 case
            blocks = [[]]
        return self.__class__(scs.bmat(blocks, format='csr', dtype=self.dtype))

    def find(self, condition):
        """Indices of current array with True condition."""
        # size instead of shape as we just need one condition for each element
        if condition.size != self.size:
            raise ValueError("condition size must be {:}".format(self.size))
        return map(list, scs.find(condition.tocsr()))[:2]

    def bincount(self):
        raise NotImplementedError(
            "Bincount not implemented for sparse! Convert to dense first.")

    def sort(self, axis=-1):
        """sort array in places"""
        if axis == 1 or axis == -1:
            for i in xrange(self.shape[0]):
                row = self[i, :].todense()
                row.sort(axis=1)
                self[i, :] = row
        elif axis == 0:
            for i in xrange(self.shape[1]):
                column = self[:, i].todense()
                column.sort(axis=0)
                self[:, i] = column
        else:
            raise ValueError("wrong sorting axis.")

    def argsort(self, axis=-1):
        """
        Returns the indices that would sort an array.
        If possible is better if you use sort function 
        axis= -1 order based on last axis (which in sparse matrix is 1 horizontal)
        """
        # for all element of chosen axis
        if axis is None:
            array = self.ravel()
            axis_elem_num = 1  # order for column
        else:
            array = self
            if axis == 1 or axis == -1:
                axis = 1
                axis_elem_num = array.shape[0]  # order for row
            elif axis == 0:
                axis_elem_num = array.shape[1]  # order for column
            else:
                raise ValueError(
                    "wrong axis parameter in argsort function for sparse data")

        index_matrix = Cdense().zeros(array.shape, dtype=int)

        axis_element = None
        for i in xrange(axis_elem_num):

            if axis == 1 or axis == -1 or axis is None:
                axis_element = array[i, :]  # order for row
            elif axis == 0:
                axis_element = array[:, i]  # order for column

            # argsort of current axis element
            sorted_data_idx = Cdense(
                axis_element.todense()).argsort(axis=axis, kind='quicksort')

            if axis == 1 or axis == -1 or axis is None:
                index_matrix[i, :] = sorted_data_idx[0, :]  # order for row
            elif axis == 0:
                index_matrix[:, i] = sorted_data_idx[:, 0]  # order for column

        return index_matrix.ravel() if axis is None else index_matrix

    def argmin(self, axis=None):
        """Indices of the minimum values along an axis.

        Parameters
        ----------
        axis : int, None, optional
            If None (default), array is flattened before computing
            index, otherwise the specified axis is used.

        Returns
        -------
        index : int, Cdense
            Scalar with index of the minimum value for flattened array or
            Cdense with indices along the given axis.

        Notes
        -----
        In case of multiple occurrences of the minimum values, the
        indices corresponding to the first occurrence are returned.

        Examples
        --------
        >>> from secml.array import Csparse

        >>> Csparse([-1, 0, 3]).argmin()
        Cdense([0])

        >>> Csparse([[-1, 0],[4, 3]]).argmin(axis=0)  # We return the index of minimum for each row
        Cdense([[0, 0]])

        >>> Csparse([[-1, 0],[4, 3]]).argmin(axis=1)  # We return the index of maximum for each column
        Cdense([[0],
               [1]])

        """
        if self.size == 0:
            raise ValueError("attempt to get argmin of an empty sequence")

        # Preparing data
        if axis is None or axis == 1 or axis == -1:
            array = self.ravel() if axis is None else self
            axis_elem_num = array.shape[0]  # min for row
            this_indices = array.nnz_row_indices
            other_axis_indices = array.nnz_column_indices
        elif axis == 0:
            array = self
            axis_elem_num = array.shape[1]  # min for column
            this_indices = array.nnz_column_indices
            other_axis_indices = array.nnz_row_indices
        else:
            raise ValueError("{:} is not a valid axis.")

        index_matrix = Cdense.zeros(axis_elem_num, dtype=array.dtype)

        for i in xrange(axis_elem_num):

            # search minimum between non zero element for current row/column
            i_indices = this_indices.find(this_indices == i)[1]

            if len(i_indices) != 0:
                # there is at least one element different from zero
                current_elem = array._data.data[i_indices]
                elem_min_idx = current_elem.argmin()
                nnz_min = current_elem[elem_min_idx]
                nnz_min_idx = other_axis_indices[0, i_indices][0, elem_min_idx]
            else:
                nnz_min = 0
                nnz_min_idx = 0

            # if min found is greater than zero...
            if nnz_min > 0:

                use_axis = 1 if axis is None or axis == 1 or axis == -1 else 0

                # ...at least a zero in current row/column, zero is the min
                if len(i_indices) < array.shape[use_axis]:

                    if use_axis == 1:  # order for row
                        all_i_element = array[i, :].todense()
                    elif use_axis == 0:  # order for column
                        all_i_element = array.T[i, :].todense()
                    else:
                        raise ValueError("{:} is not a valid axis.")

                    nnz_min_idx = all_i_element.find(all_i_element == 0)[1][0]

            index_matrix[0, i] = nnz_min_idx

        return index_matrix.ravel()[0, 0] if axis is None else \
            [index_matrix.atleast_2d() if axis == 0 else
             index_matrix.atleast_2d().T][0]

    def argmax(self, axis=None):
        """Indices of the maximum values along an axis.

        Parameters
        ----------
        axis : int, None, optional
            If None (default), array is flattened before computing
            index, otherwise the specified axis is used.

        Returns
        -------
        index : int, Cdense
            Scalar with index of the maximum value for flattened array or
            Cdense with indices along the given axis.

        Notes
        -----
        In case of multiple occurrences of the maximum values, the
        indices corresponding to the first occurrence are returned.

        Examples
        --------
        >>> from secml.array import Csparse

        >>> Csparse([-1, 0, 3]).argmax()
        Cdense([2])

        >>> Csparse([[-1, 0],[4, 3]]).argmax(axis=0)  # We return the index of minimum for each row
        Cdense([[1, 1]])

        >>> Csparse([[-1, 0],[4, 3]]).argmax(axis=1)  # We return the index of maximum for each column
        Cdense([[1],
               [0]])

        """
        if self.size == 0:
            raise ValueError("attempt to get argmin of an empty sequence")

        # Preparing data
        if axis is None or axis == 1 or axis == -1:
            array = self.ravel() if axis is None else self
            axis_elem_num = array.shape[0]  # max for row
            this_indices = array.nnz_row_indices
            other_axis_indices = array.nnz_column_indices
        elif axis == 0:
            array = self
            axis_elem_num = array.shape[1]  # max for column
            this_indices = array.nnz_column_indices
            other_axis_indices = array.nnz_row_indices
        else:
            raise ValueError("{:} is not a valid axis.")

        index_matrix = Cdense.zeros(axis_elem_num, dtype=array.dtype)

        for i in xrange(axis_elem_num):

            # search maximum between non zero element for current row/column
            i_indices = this_indices.find(this_indices == i)[1]

            if len(i_indices) != 0:
                # there is at least one element different from zero
                current_elem = array._data.data[i_indices]
                elem_max_idx = current_elem.argmax()
                nnz_max = current_elem[elem_max_idx]
                nnz_max_idx = other_axis_indices[0, i_indices][0, elem_max_idx]
            else:
                nnz_max = 0
                nnz_max_idx = 0

            # if max found is below zero...
            if nnz_max < 0:

                use_axis = 1 if axis is None or axis == 1 or axis == -1 else 0

                # ...at least a zero in current row/column, zero is the max
                if len(i_indices) < array.shape[use_axis]:

                    if use_axis == 1:  # order for row
                        all_i_element = array[i, :].todense()
                    elif use_axis == 0:  # order for column
                        all_i_element = array.T[i, :].todense()
                    else:
                        raise ValueError("{:} is not a valid axis.")

                    nnz_max_idx = all_i_element.find(all_i_element == 0)[1][0]

            index_matrix[0, i] = nnz_max_idx

        return index_matrix.ravel()[0, 0] if axis is None else \
            [index_matrix.atleast_2d() if axis == 0 else
             index_matrix.atleast_2d().T][0]

    def nanargmax(self, axis=None):
        raise NotImplementedError

    def nanargmin(self, axis=None):
        raise NotImplementedError

    def logical_and(self, y):
        """Element-wise logical AND of array elements.

        Compare two arrays and returns a new array containing
        the element-wise logical AND.

        Parameters
        ----------
        y : csr_Csparse or array_like
            The array like object holding the elements to compare
            current array with. Must have the same shape of first
            array.

        Returns
        -------
        out_and : csr_Csparse or bool
            The element-wise logical AND between first array and y.

        Examples
        --------
        >>> from secml.array import Csparse

        >>> Csparse([[-1,0],[2,0]]).logical_and(Csparse([[2,-1],[2,-1]])).todense()
        Cdense([[ True, False],
               [ True, False]], dtype=bool)

        >>> Csparse([-1]).logical_and(Csparse([2])).todense()
        Cdense([[ True]], dtype=bool)

        >>> array = Csparse([1,0,2,-1])
        >>> (array > 0).logical_and(array < 2).todense()
        Cdense([[ True, False, False, False]], dtype=bool)

        """
        if self.shape != y.shape:
            raise ValueError("array to compare must be {:}".format(self.shape))

        and_result = self.__class__(self.shape, dtype=bool)

        for el_idx, el in enumerate(self._data.data):
            # Get indices of current element
            this_elem_row = self.nnz_row_indices[0, el_idx]
            this_elem_col = self.nnz_column_indices[0, el_idx]
            # Check is second array has an element in the same position
            y_same_bool = (y.nnz_row_indices == this_elem_row).logical_and(
                y.nnz_column_indices == this_elem_col)
            if y_same_bool.any() == True:  # found an element in same position!
                same_position_val = int(y._data.data[y_same_bool.toarray()])
                if np.logical_and(el, same_position_val):
                    and_result[this_elem_row, this_elem_col] = True

        return and_result

    def logical_or(self, y):
        """Element-wise logical OR of array elements.

        Compare two arrays and returns a new array containing
        the element-wise logical OR.

        Parameters
        ----------
        y : csr_Csparse or array_like
            The array like object holding the elements to compare
            current array with. Must have the same shape of first
            array.

        Returns
        -------
        out_and : csr_Csparse or bool
            The element-wise logical OR between first array and y.
         

        Examples
        --------
        >>> from secml.array import Csparse

        >>> Csparse([[-1,0],[2,0]]).logical_or(Csparse([[2,0],[2,-1]])).todense()
        Cdense([[ True, False],
               [ True,  True]], dtype=bool)

        >>> Csparse([False]).logical_and(Csparse([False])).todense()
        Cdense([[False]], dtype=bool)

        >>> array = Csparse([1,0,2,-1])
        >>> (array > 0).logical_or(array < 2).todense()
        Cdense([[ True,  True,  True,  True]], dtype=bool)

        """
        if self.shape != y.shape:
            raise ValueError("array to compare must be {:}".format(self.shape))

        or_matrix = self.astype(bool)
        # Be sure there are only non-zeros in 2nd array
        y.eliminate_zeros()
        # add True for each non-zero element of second array
        or_matrix[[y.nnz_row_indices.tolist(),
                   y.nnz_column_indices.tolist()]] = True

        return or_matrix

    def logical_not(self):
        """Element-wise logical NOT of array elements."""
        # Be sure there are only non-zeros in array
        self.eliminate_zeros()
        # Storing indices of elements will be zeros
        old_nnz_row_indices = self.nnz_row_indices.tolist()
        old_nnz_column_indices = self.nnz_column_indices.tolist()

        # Create an array full of Trues (VERY expensive!)
        new_not = self.__class__(Cdense.ones(*self.shape)).astype(bool)
        # All old nonzeros should be zeros!
        new_not[[old_nnz_row_indices, old_nnz_column_indices]] = False
        # Eliminate newly created zeros from internal csr_matrix structures
        new_not.eliminate_zeros()

        return new_not

    # -------------------------------- #
    # # # # # # CLASSMETHODS # # # # # #
    # ---------------------------------#

    @classmethod
    def eye(cls, n_rows, n_cols=None, k=0, dtype=float):
        """Return an array of desired dimension with ones on the diagonal and zeros elsewhere.
        See scipy.sparse.eye for more informations.

        Parameters
        ----------
        n_rows : number of rows for output array, integer.
        n_cols : number of columns in the output. If None, defaults to n_rows.
        k : index of the diagonal. 0 (the default) refers to the main diagonal,
            a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
        dtype : datatype of array data.

        Returns
        -------
        Sparse array of desired shape with ones on the diagonal and zeros elsewhere.

        Examples
        --------
        >>> from secml.array import Csparse
        >>> array = Csparse.eye(2)
        >>> print array  # doctest: +SKIP
        (0, 0)	1.0
        (1, 1)	1.0
        >>> print array.shape
        (2, 2)

        >>> array = Csparse.eye(2, k=1, dtype=int)
        >>> print array  # doctest: +SKIP
        (0, 1)	1
        >>> print array.shape
        (2, 2)

        """
        return cls(scs.eye(n_rows, n_cols, k=k, dtype=dtype, format='csr'))

    @classmethod
    def rand(cls, n_rows, n_cols, density=0.01):
        """Wrapper for scipy.sparse.rand.

        Creates a random sparse array of [0, 1] floats
        with input density and shape.
        Density equal to one means a dense matrix,
        density of 0 means a matrix with no non-zero items.

        """
        return cls(scs.rand(n_rows, n_cols, density=density, format='csr'))

    @classmethod
    def concatenate(cls, array1, array2, axis=1):
        """Concatenate a sequence of arrays along the given axis."""
        if not isinstance(array1, cls) or not isinstance(array2, cls):
            raise TypeError(
                "both arrays to concatenate must be {:}".format(cls))

        if axis is not None:
            if array1.shape[abs(axis - 1)] != array2.shape[abs(axis - 1)]:
                raise ValueError("all the input array dimensions except for "
                                 "the concatenation axis must match exactly.")
        else:  # axis is None, both arrays should be ravelled
            array1 = array1.ravel()
            array2 = array2.ravel()
            axis = 1  # Simulate an horizontal concatenation

        if axis == 1:  # horizontal concatenation
            array1 = array1.T
            array2 = array2.T

        # Use vertical concatenation in all cases
        data = np.append(array1._data.data, array2._data.data)
        indices = np.append(array1._data.indices, array2._data.indices)
        indptr = np.append(array1._data.indptr,
                           array2._data.indptr[1:] + array1._data.indptr[-1])
        new_array = cls(
            (data, indices, indptr),
            shape=(array1.shape[0] + array2.shape[0], array1.shape[1]))

        return new_array.T if axis == 1 else new_array

    @classmethod
    def comblist(cls, list_of_list, dtype=float):
        """Return the norm of store data."""
        raise NotImplementedError
