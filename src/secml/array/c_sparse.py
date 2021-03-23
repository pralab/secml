"""
.. module:: CSparse
   :synopsis: Wrapper of `scipy.sparse` sparse matrices

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
import scipy.sparse as scs
from scipy.sparse.linalg import inv, norm
import numpy as np

from secml.array.c_array_interface import _CArrayInterface

from secml.array.c_dense import CDense

from secml.core.type_utils import is_ndarray, is_list_of_lists, \
    is_list, is_tuple, is_slice, is_scalar, is_int, is_bool
from secml.core.constants import inf


def _expand_nnz_bool(array, nnz_val):
    """Convert a bool condition evaluated on `array.nnz_val` to full array shape.

    Parameters
    ----------
    array : CSparse
        Reference sparse array.
    nnz_val : CDense
        Dense array of bool, with True/False values of the desired
        function evaluated on `nnz_data`

    Returns
    -------
    CSparse
        Convert the input dense array of bool evaluated on `nnz_data` to
        a sparse array of bool, with True/False in the positions
        corresponding to the `nnz_data`.

    """
    out = CSparse.empty(shape=array.shape, dtype=bool)

    if array.size == 0:  # Empty array, return empty
        return out

    if len(nnz_val.shape) != 2:  # nnz_val must be (1, N)
        raise RuntimeError("unexpected shape {:}".format(nnz_val.shape))

    # Get the indices of array where the True values are
    nnz_val_true_idx = nnz_val.find(nnz_val)[1]
    true_idx = [[e[i] for i in nnz_val_true_idx] for e in array.nnz_indices]

    out[true_idx] = True

    return out


def _shape_atleast_2d(shape):
    """Convert input shape to a two-dimensional shape.

    Parameters
    ----------
    shape : int or tuple or other
        Shape to be converted.

    Returns
    -------
    tuple
        Two-dimensional shape.
        If shape is an int, `(1, shape)` will be returned.
        If shape is a tuple and `len(shape) == 1`,
        `(1, shape[0])` will be returned.
        Otherwise, input shape will be returned as is.

    """
    if is_int(shape):
        shape = (1, shape)
    if is_tuple(shape) and len(shape) < 2:
        shape = (1, shape[0])
    return shape


class CSparse(_CArrayInterface):
    """Sparse array. Encapsulation for scipy.sparse.csr_matrix."""
    __slots__ = '_data'  # CSparse has only one slot for the scs.csr_matrix

    def __init__(self, data=None, dtype=None, copy=False, shape=None):
        """Sparse matrix initialization."""
        # Not implemented operators return NotImplemented
        if data is NotImplemented:
            raise TypeError("operator not implemented")
        if isinstance(data, CDense):
            self._input_shape = data.input_shape  # Propagate original shape
            data = data.tondarray()  # np.ndarray from CDense
        elif isinstance(data, self.__class__):
            self._input_shape = data.input_shape  # Propagate original shape
            data = data.tocsr()  # scs.csr_matrix from CSparse
        else:  # Other inputs... just need to initialize the input shape
            self._input_shape = None
        # Reshaping is not supported for csr_matrix so we need few hacks
        # We don't use shape when creating the buffer, but we reshape later
        # Scipy >= 1.4, shape must be two dimensional
        newshape = _shape_atleast_2d(shape)
        if not is_tuple(data):
            shape = None  # This is the shape passed to scs.csr_matrix()
            if not scs.issparse(data):  # ndarray, list of lists
                data = np.array(data, ndmin=1)  # Common dense format
                # Store the shape of input data (if not previously propagated)
                # before any further reshaping
                if self.input_shape is None:
                    self._input_shape = data.shape
                # If input data has > 2 dims, reshape to 2 dims
                if data.ndim > 2:
                    data = data.reshape(data.shape[0], -1)
            else:  # For sparse arrays we directly store shape (always ndim=2)
                # ... but again only if not previously propagated
                if self.input_shape is None:
                    self._input_shape = data.shape
        else:  # For special scipy init, shape should be passed to `csr_matrix`
            shape = newshape
            self._input_shape = data  # data is also the input shape
        self._data = scs.csr_matrix(data, shape, dtype, copy)
        # Now we reshape the array if needed (not available for scs.csr_matrix)
        if newshape is not None and newshape != self.shape:
            self._data = self.reshape(newshape)._data

    # ------------------------------ #
    # # # # # # PROPERTIES # # # # # #
    # -------------------------------#

    @property
    def shape(self):
        return self._data.shape

    @property
    def input_shape(self):
        """Original shape of input data, tuple of ints."""
        return self._input_shape

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
        """Number of non-zero values in the array."""
        return self.get_nnz()

    @property
    def nnz_indices(self):
        """Return a list of list that contain index of non zero elements."""
        return list(map(list, self.tocsr().nonzero()))

    @property
    def nnz_data(self):
        """Return non zero elements."""
        if len(self.nnz_indices[0]) == 0:
            return CDense([])
        return self[self.nnz_indices].todense()

    @property
    def T(self):
        return self.transpose()

    @property
    def is_vector_like(self):
        """True if array is vector-like.

        An array is vector-like when shape[0] == 1.

        Returns
        -------
        bool
            True if array is vector-like.

        """
        if self.shape[0] == 1:
            return True
        else:
            return False

    # --------------------------- #
    # # # # # # CASTING # # # # # #
    # ----------------------------#

    def tondarray(self, order=None, shape=None):
        """Convert csr_matrix to ndarray.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to store multidimensional data in C (row-major) or
            Fortran (column-major) order in memory. The default is 'None',
            indicating the NumPy default of C-ordered.
        shape : int or tuple of ints, optional
            The new shape for the output data.
            Reshape is performed after casting.

        """
        out = self._data.toarray(order)
        if shape is not None:
            return out.reshape(shape)
        return out

    def _toscs(self, scs_format, shape=None):
        """Return data as input scipy.scs format.

        Parameters
        ----------
        scs_format : str
            Scipy sparse format.
        shape : tuple of ints, optional
            The new shape for the output data. Must be 2-Dimensional.
            Reshape is performed after casting.

        """
        out = getattr(self._data, 'to{:}'.format(scs_format))()
        if shape is not None:
            if not is_tuple(shape) or len(shape) != 2:
                # TODO: ERROR IS PROPERLY RAISED IN SCIPY > 1.4
                raise ValueError('matrix shape must be two-dimensional')
            # output of scipy.reshape not necessarily of the same format
            return getattr(out.reshape(shape), 'to{:}'.format(scs_format))()
        return out

    def tocsr(self, shape=None):
        """Return data as csr_matrix.

        Parameters
        ----------
        shape : tuple of ints, optional
            The new shape for the output data. Must be 2-Dimensional.
            Reshape is performed after casting.

        """
        return self._toscs('csr', shape=shape)

    def tocoo(self, shape=None):
        """Return data as coo_matrix.

        Parameters
        ----------
        shape : tuple of ints, optional
            The new shape for the output data. Must be 2-Dimensional.
            Reshape is performed after casting.

        """
        return self._toscs('coo', shape=shape)

    def tocsc(self, shape=None):
        """Return data as csc_matrix.

        Parameters
        ----------
        shape : tuple of ints, optional
            The new shape for the output data. Must be 2-Dimensional.
            Reshape is performed after casting.

        """
        return self._toscs('csc', shape=shape)

    def todia(self, shape=None):
        """Return data as dia_matrix.

        Parameters
        ----------
        shape : tuple of ints, optional
            The new shape for the output data. Must be 2-Dimensional.
            Reshape is performed after casting.

        """
        return self._toscs('dia', shape=shape)

    def todok(self, shape=None):
        """Return data as dok_matrix.

        Parameters
        ----------
        shape : tuple of ints, optional
            The new shape for the output data. Must be 2-Dimensional.
            Reshape is performed after casting.

        """
        return self._toscs('dok', shape=shape)

    def tolil(self, shape=None):
        """Return data as lil_matrix.

        Parameters
        ----------
        shape : tuple of ints, optional
            The new shape for the output data. Must be 2-Dimensional.
            Reshape is performed after casting.

        """
        return self._toscs('lil', shape=shape)

    def tolist(self, shape=None):
        """Return data as list.

        Parameters
        ----------
        shape : int or tuple of ints, optional
            The new shape for the output data. The array is converted to
            ndarray first, then reshaping is performed.

        """
        return self.todense().tolist(shape=shape)

    def todense(self, order=None):
        """Return data as CDense."""
        obj = CDense(self.tondarray(order))
        obj._input_shape = self.input_shape
        return obj

    def _tocoo_or_tocsr(self):
        """Return data as coo_matrix if data is not as csr_matrix,
        return csr_matrix otherwise."""
        if self._data.getformat() != 'csr':
            return self.tocoo()
        return self.tocsr()

    def _buffer_to_builtin(self, data):
        """Convert data buffer to built-in arrays"""
        if isinstance(data, CDense):  # Extract np.ndarray
            return data.tondarray()
        elif isinstance(data, self.__class__):  # Extract scs.csr_matrix
            return data.tocsr()
        else:
            return data

    # ---------------------------- #
    # # # # # # INDEXING # # # # # #
    # -----------------------------#

    def _check_index(self, idx):
        """Consistency checks for __getitem__ and __setitem__ functions.

        1 index can be used for 2D arrays with shape[0] == 1 (vector-like).

        Parameters
        ----------
        idx : object
            - CDense, CSparse boolean masks
              Number of rows should be equal to 2.
            - List of lists (output of `find_2d` method).
              Number of elements should be equal to 2.
            - tuple of 2 or more elements. Any of the following:
                - CDense, CSparse
                - Iterable built-in types (list, slice).
                - Atomic built-in types (int, bool).
                - Numpy atomic types (np.integer, np.bool_).
            - for vector-like arrays, one element between the above ones.

        """
        if isinstance(idx, CDense) or isinstance(idx, CSparse):

            # Boolean mask
            if idx.dtype.kind == 'b':

                # Boolean masks must be 2-Dimensional
                if idx.ndim == 1:
                    idx = idx.atleast_2d()

                # Convert indices to built-in arrays
                idx = idx.tondarray() if isinstance(idx, CDense) else idx
                idx = idx.tocsr() if isinstance(idx, CSparse) else idx

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
            idx = (np.asarray([0]), idx.tondarray())

            # Matlab-like indexing
            idx = np.ix_(*idx)

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

            # Matlab-like indexing
            idx = np.ix_(*idx)

        elif is_list(idx):
            # Check if array is vector-like
            if self.shape[0] != 1:
                raise IndexError("vector-like indexing is only applicable "
                                 "to arrays with shape[0] == 1.")

            # Empty lists are converted to float by numpy,
            # special handling needed
            if len(idx) == 0:
                idx = np.asarray(idx, dtype=int)
            else:  # Otherwise we leave np decide
                idx = np.asarray(idx)

            # Fake 2D index. Use ndarrays to mimic Matlab-like indexing
            idx = (np.asarray([0]), idx)

            # Check the size of any boolean array inside tuple
            self._check_index_bool(idx)

            # Matlab-like indexing
            idx = np.ix_(*idx)

        elif is_slice(idx):
            # Check if array is vector-like
            if self.shape[0] != 1:
                raise IndexError("vector-like indexing is only applicable "
                                 "to arrays with shape[0] == 1.")

            # Fake index for row. Slice for columns is fine
            idx = (0, idx)

            # For fast column slicing we use csc. Also, for slices with
            # step != 1 the result will actually be wrong using csr format
            self._data = self._data.tocsc()

        elif isinstance(idx, tuple):

            # Tuple will be now transformed to be managed directly by numpy

            idx_list = [idx[0], idx[1]]  # Use list to change indices type

            for e_i, e in enumerate(idx_list):
                # Check each tuple element and convert to ndarray
                if isinstance(e, CDense) or isinstance(e, CSparse):
                    if not e.is_vector_like:
                        raise IndexError("invalid index shape")
                    idx_list[e_i] = e.tondarray().ravel()
                    # Check the size of any boolean array inside tuple
                    t = [None, None]  # Fake index for booleans check
                    t[e_i] = idx_list[e_i]
                    self._check_index_bool(tuple(t))

                elif is_list(e):
                    # Empty lists are converted to float by numpy,
                    # special handling needed
                    if len(e) == 0:
                        idx_list[e_i] = np.asarray(e, dtype=int)
                    else:  # Otherwise we leave np decide
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

                elif is_slice(e):  # slice excluded (keep slice)
                    idx_list[e_i] = e
                    if e_i == 1:
                        # For fast column slicing we use csc.
                        # Also, for slices with step != 1 the result
                        # will actually be wrong using csr format
                        self._data = self._data.tocsc()

                else:
                    raise TypeError("{:} should not be used for "
                                    "CSparse indexing.".format(type(e)))

            # Converting back to tuple
            idx = tuple(idx_list)

            # Matlab-like indexing
            if all(is_ndarray(elem) for elem in idx):
                idx = np.ix_(*idx)

        else:
            # No other object is accepted for CSparse indexing
            raise TypeError("{:} should not be used for "
                            "CSparse indexing.".format(type(idx)))

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

        # Ready for scipy. We cast the result of scipy's getitem as the
        # original dtype is not kept sometimes (especially for empty arrays)
        return self.__class__(self._data.__getitem__(idx), dtype=self.dtype)

    def __setitem__(self, idx, value):
        """Redefinition of the get (brackets) operator."""
        # Check for setitem value
        if isinstance(value, CDense):
            if value.is_vector_like:
                if value.ndim > 1:
                    # vector-like arrays of 2 or more dims to vectors
                    # in order to always perform the set operation correctly
                    value = value.ravel()
                elif is_list_of_lists(idx):
                    # Scipy v1.3+, list of list indexing returns 1-D, ravel
                    # input if vector-like (otherwise error should be raised)
                    value = value.ravel()
            value = value.tondarray()
        elif isinstance(value, CSparse):
            value = value.tocsr()
        elif not (is_scalar(value) or is_bool(value)):
            raise TypeError("{:} cannot be used for setting "
                            "a CSparse.".format(type(value)))

        # Check index for all other cases
        idx = self._check_index(idx)

        # We use lil format for efficient changing of sparsity structure
        self._data = self._data.tolil()

        # The tuple can now be managed directly by scipy
        self._data.__setitem__(idx, value)

        # Convert the internal buffer back to csr format
        self._data = self._data.tocsr()

        # Cleaning array after setting
        self.eliminate_zeros()

    # ------------------------------------ #
    # # # # # # SYSTEM OVERLOADS # # # # # #
    # -------------------------------------#

    def _broadcast_other(self, other):
        """Broadcast `other` to have the same shape of self.

        This only performs left side single row/column broadcast.

        Parameters
        ----------
        other : CSparse
            Array to be broadcasted.

        Returns
        -------
        CSparse
            Broadcasted array.

        """
        if self.shape != other.shape:
            if self.shape[0] == other.shape[0]:  # Equal number of rows
                if other.shape[1] == 1:
                    other = other.repmat(1, self.shape[1])

            elif self.shape[1] == other.shape[1]:  # Equal number of cols
                if other.shape[0] == 1:
                    other = other.repmat(self.shape[0], 1)

        return other

    def __add__(self, other):
        """Element-wise addition.

        Parameters
        ----------
        other : CSparse or Cdense
            Element to add to current array.

        Returns
        -------
        array : CSparse or CDense
            If input is a CSparse, a CSparse will be returned.
            If input is a Cdense, a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            if other == 0:
                return self.deepcopy()
            raise NotImplementedError(
                "adding a nonzero scalar or a boolean True to a "
                "sparse array is not supported. Convert to dense if needed.")
        elif isinstance(other, CSparse):  # Sparse + Sparse = Sparse
            # Scipy does not support broadcast natively
            other = self._broadcast_other(other)
            return self.__class__(self._data.__add__(other.tocsr()))
        elif isinstance(other, CDense):  # Sparse + Dense = Dense
            if other.size == 1:  # scalar-like
                raise NotImplementedError(
                    "adding an array of size one to a sparse array "
                    "is not supported. Convert to dense if needed.")
            else:  # direct operation or broadcast
                return CDense(self._data.__add__(other.tondarray()))
        else:
            return NotImplemented

    def __radd__(self, other):
        """Element-wise (inverse) addition."""
        return self.__add__(other)

    def __sub__(self, other):
        """Element-wise subtraction.

        Parameters
        ----------
        other : CSparse or CDense
            Element to subtraction to current array.

        Returns
        -------
        array : CSparse or Cdense
            If input is a CSparse, a CSparse will be returned.
            If input is a Cdense, a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            if other == 0:
                return self.deepcopy()
            raise NotImplementedError(
                "subtracting a nonzero scalar or a boolean True from a "
                "sparse array is not supported. Convert to dense if needed.")
        elif isinstance(other, CSparse):  # Sparse - Sparse = Sparse
            # Scipy does not support broadcast natively
            other = self._broadcast_other(other)
            return self.__class__(self._data.__sub__(other.tocsr()))
        elif isinstance(other, CDense):  # Sparse - Dense = Dense
            if other.size == 1:  # scalar-like
                raise NotImplementedError(
                    "subtracting an array of size one from a sparse array "
                    "is not supported. Convert to dense if needed.")
            else:  # direct operation or broadcast
                return CDense(self._data.__sub__(other.tondarray()))
        else:
            return NotImplemented

    def __rsub__(self, other):
        """Element-wise (inverse) subtraction."""
        if is_scalar(other) or is_bool(other):
            if other == 0:
                return -self.deepcopy()
            raise NotImplementedError(
                "subtracting a sparse array from a nonzero scalar or from "
                "a boolean True is not supported. Convert to dense if needed.")
        else:
            return NotImplemented

    def __mul__(self, other):
        """Element-wise product.

        Parameters
        ----------
        other : CSparse or CDense or scalar or bool
            Element to multiply to current array. If an array, element-wise
            product will be performed. If scalar or boolean, the element
            will be multiplied to each array element.

        Returns
        -------
        array : CSparse
            Array after product.

        """
        if is_scalar(other) or is_bool(other) or \
                isinstance(other, (CSparse, CDense)):  # Always Sparse
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
        array : CSparse
            Array after product.

        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """Element-wise true division.

        Parameters
        ----------
        other : CSparse or CDense or scalar or bool
            Element to divide to current array. If an array, element-wise
            division will be performed. If scalar or boolean, the element
            will be divided to each array element.

        Returns
        -------
        array : CSparse or Cdense
            Array after division. CSparse if other is a scalar or bool,
            Cdense otherwise.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__truediv__(other))
        elif isinstance(other, CSparse):  # Sparse / Sparse = Dense
            # Scipy does not support broadcast natively
            other = self._broadcast_other(other)
            return CDense(self._data.__truediv__(other.tocsr()))
        elif isinstance(other, CDense):  # Sparse / Dense = Dense
            # Compatible shapes, call built-in div
            return CDense(self._data.__truediv__(other.tondarray()))
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """Element-wise (inverse) true division."""
        raise NotImplementedError(
            "dividing a scalar by a sparse array is not supported")

    def __floordiv__(self, other):
        """Element-wise floor division (integral part of the quotient).

        Parameters
        ----------
        other : CSparse or scalar or bool
            Element to divided to current array. If a CDense, element-wise
            division will be performed. If scalar or boolean, the element
            will be divided to each array element.

        Returns
        -------
        array : CSparse
            Array after division.

        """
        # Scipy does not implement floor division
        out_truediv = self.__truediv__(other)
        if out_truediv is NotImplemented:
            return NotImplemented
        else:  # Return the integer part of the truediv result
            return out_truediv.floor()

    def __rfloordiv__(self, other):
        """Element-wise (inverse) floor division."""
        raise NotImplementedError(
            "dividing a scalar by a sparse array is not supported")

    def __abs__(self):
        """Returns array elements without sign.

        Returns
        -------
        array : CSparse
            Array with the corresponding elements without sign.

        """
        return self.__class__(self._data.__abs__())

    def __neg__(self):
        """Returns array elements with negated sign.

        Returns
        -------
        array : CDense
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
        array : CSparse
            Array after power.

        """
        if is_scalar(power) or is_bool(power):
            if power == 0:
                raise NotImplementedError(
                    "using zero or a boolean False as power is not supported "
                    "for sparse arrays. Convert to dense if needed.")
            x = self.tocsr()  # self.__class__ expects a csr
            # indices/indptr must passed as copies (pow creates new data)
            return self.__class__((pow(x.data, power), x.indices, x.indptr),
                                  shape=x.shape, copy=True)
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
        other : CSparse or CDense or scalar or bool
            Element to be compared. If an array, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CSparse or Cdense
            If input is a CSparse or a scalar or a bool,
            a CSparse will be returned. If input is a Cdense,
            a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__eq__(other))
        elif isinstance(other, CSparse):  # Sparse == Sparse = Sparse
            # Scipy does not support broadcast natively
            other = self._broadcast_other(other)
            return self.__class__(self._data.__eq__(other.tocsr()))
        elif isinstance(other, CDense):  # Sparse == Dense = Dense
            return CDense(self._data.__eq__(other.tondarray()))
        else:
            return NotImplemented

    def __lt__(self, other):
        """Element-wise < operator.

        Parameters
        ----------
        other : CSparse or CDense or scalar or bool
            Element to be compared. If an array, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CSparse or Cdense
            If input is a CSparse or a scalar or a bool,
            a CSparse will be returned. If input is a Cdense,
            a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__lt__(other))
        elif isinstance(other, CSparse):  # Sparse < Sparse = Sparse
            # Scipy does not support broadcast natively
            other = self._broadcast_other(other)
            return self.__class__(self._data.__lt__(other.tocsr()))
        elif isinstance(other, CDense):  # Sparse < Dense = Dense
            return CDense(self._data.__lt__(other.tondarray()))
        else:
            return NotImplemented

    def __le__(self, other):
        """Element-wise <= operator.

        Parameters
        ----------
        other : CSparse or CDense or scalar or bool
            Element to be compared. If an array, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CSparse or Cdense
            If input is a CSparse or a scalar or a bool,
            a CSparse will be returned. If input is a Cdense,
            a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__le__(other))
        elif isinstance(other, CSparse):  # Sparse <= Sparse = Sparse
            # Scipy does not support broadcast natively
            other = self._broadcast_other(other)
            return self.__class__(self._data.__le__(other.tocsr()))
        elif isinstance(other, CDense):  # Sparse <= Dense = Dense
            return CDense(self._data.__le__(other.tondarray()))
        else:
            return NotImplemented

    def __gt__(self, other):
        """Element-wise > operator.

        Parameters
        ----------
        other : CSparse or CDense or scalar or bool
            Element to be compared. If an array, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CSparse or Cdense
            If input is a CSparse or a scalar or a bool,
            a CSparse will be returned. If input is a Cdense,
            a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__gt__(other))
        elif isinstance(other, CSparse):  # Sparse > Sparse = Sparse
            # Scipy does not support broadcast natively
            other = self._broadcast_other(other)
            return self.__class__(self._data.__gt__(other.tocsr()))
        elif isinstance(other, CDense):  # Sparse > Dense = Dense
            return CDense(self._data.__gt__(other.tondarray()))
        else:
            return NotImplemented

    def __ge__(self, other):
        """Element-wise >= operator.

        Parameters
        ----------
        other : CSparse or CDense or scalar or bool
            Element to be compared. If an array, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CSparse or CDense
            If input is a CSparse or a scalar or a bool,
            a CSparse will be returned. If input is a Cdense,
            a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__ge__(other))
        elif isinstance(other, CSparse):  # Sparse >= Sparse = Sparse
            # Scipy does not support broadcast natively
            other = self._broadcast_other(other)
            return self.__class__(self._data.__ge__(other.tocsr()))
        elif isinstance(other, CDense):  # Sparse >= Dense = Dense
            return CDense(self._data.__ge__(other.tondarray()))
        else:
            return NotImplemented

    def __ne__(self, other):
        """Element-wise != operator.

        Parameters
        ----------
        other : CSparse or Cdense or scalar or bool
            Element to be compared. If an array, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        array : CSparse or CDense
            If input is a CSparse or a scalar or a bool,
            a CSparse will be returned. If input is a Cdense,
            a Cdense will be returned.

        """
        if is_scalar(other) or is_bool(other):
            return self.__class__(self._data.__ne__(other))
        elif isinstance(other, CSparse):  # Sparse != Sparse = Sparse
            # Scipy does not support broadcast natively
            other = self._broadcast_other(other)
            return self.__class__(self._data.__ne__(other.tocsr()))
        elif isinstance(other, CDense):  # Sparse != Dense = Dense
            return CDense(self._data.__ne__(other.tondarray()))
        else:
            return NotImplemented

    def __bool__(self):
        """Manage 'and' and 'or' operators."""
        return bool(self._data)

    def __iter__(self):
        """Yields array elements in raster-scan order."""
        for row_id in range(self.shape[0]):
            for column_id in range(self.shape[1]):
                yield self[row_id, column_id]

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return repr(self._data)

    # ------------------------------ #
    # # # # # # COPY UTILS # # # # # #
    # -------------------------------#

    def __copy__(self):
        """Called when copy.copy(CSparse) is called.

        Consistently with numpy.ndarray,
        this returns a DEEP COPY of current array.

        """
        out = self.__class__(self._data.copy())
        out._input_shape = self.input_shape
        return out

    def __deepcopy__(self, memo):
        """Called when copy.deepcopy(CSparse) is called."""
        out = self.__class__(self._data.copy())
        out._input_shape = self.input_shape
        memo[id(self)] = out
        return out

    # ----------------------------- #
    # # # # # # SAVE/LOAD # # # # # #
    # ------------------------------#

    def save(self, datafile, overwrite=False):
        """Save array data into plain text file.

        Data is stored preserving original data type.

        The default encoding is `utf-8`.

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

        x = self.tocsr()  # Load expects a csr_matrix

        # Flatting data to store (for sparse this results in 1 x size arrays)
        data_cndarray = CDense(x.data).reshape((1, x.data.shape[0]))
        # Converting explicitly to int as in 64 bit machines the
        # following arrays are stored with dtype == np.int32
        indices_cndarray = \
            CDense(x.indices).reshape((1, x.indices.shape[0])).astype(int)
        indptr_cndarray = \
            CDense(x.indptr).reshape((1, x.indptr.shape[0])).astype(int)

        # Error handling is managed by CDense.save()
        # file will be closed exiting from context
        with open(datafile, mode='wt+', encoding='utf-8') as fhandle:
            data_cndarray.save(fhandle)
            indices_cndarray.save(fhandle)
            indptr_cndarray.save(fhandle)
            fhandle.write(str(x.shape[0]) + " " + str(x.shape[1]))

    @classmethod
    def load(cls, datafile, dtype=float):
        """Load array data from plain text file.

        The default encoding is `utf-8`.

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
        loaded : CSparse
            Array resulting from loading, 2-dimensional.

        """
        # CDense.load() will manage IO errors
        imported_data = CDense.load(
            datafile, dtype=dtype, startrow=0, skipend=3).ravel().tondarray()
        # Indices are always integers
        imported_indices = CDense.load(
            datafile, dtype=int, startrow=1, skipend=2).ravel().tondarray()
        imported_indptr = CDense.load(
            datafile, dtype=int, startrow=2, skipend=1).ravel().tondarray()
        shape_ndarray = CDense.load(
            datafile, dtype=int, startrow=3, skipend=0).ravel().tondarray()

        return cls((imported_data, imported_indices, imported_indptr),
                   shape=(shape_ndarray[0], shape_ndarray[1]))

    # ----------------------------- #
    # # # # # # UTILITIES # # # # # #
    # ------------------------------#

    # ---------------- #
    # SHAPE ALTERATION #
    # ---------------- #

    def transpose(self):
        return self.__class__(self._data.transpose())

    def ravel(self):
        """Reshape sparse matrix to 1 x size array."""
        return self.reshape((1, self.size))

    def flatten(self):
        """Return a flattened copy of array."""
        return self.ravel().deepcopy()

    def atleast_2d(self):
        """Force array to have 2 dimensions.
        Sparse arrays are always 2 dimensional, so original array is returned.
        """
        return self.__class__(self)

    def reshape(self, newshape, order='C', copy=False):
        """Reshape the matrix using input shape (int or tuple of ints).

        Parameters
        ----------
        newshape : int or sequence of ints
            The new shape should be compatible with the original shape.
        order : {'C', 'F'}, optional
            Read the elements using this index order.
            'C' means to read and write the elements using C-like index order;
            e.g. read entire first row, then second row, etc.
            'F' means to read and write the elements using Fortran-like index
            order; e.g. read entire first column, then second column, etc.
        copy : bool, optional
            Indicates whether or not attributes of self should be copied
            whenever possible. The degree to which attributes are copied
            varies depending on the type of sparse matrix being used.

        """
        # Scipy >= 1.4, shape must be two dimensional
        newshape = _shape_atleast_2d(newshape)
        return self.__class__(
            self.tocsr().reshape(newshape, order=order, copy=copy))

    def resize(self, newshape, constant=0):
        """Return a new array with the specified shape."""
        raise NotImplementedError

    # --------------- #
    # DATA ALTERATION #
    # --------------- #

    def astype(self, dtype):
        return self.__class__(self._data.astype(dtype))

    def nan_to_num(self):
        """Replace nan with zero and inf with finite numbers."""
        # Use 'coo' for fast conversion (if not a 'csr')
        self._data = self._tocoo_or_tocsr()
        self._data.data = np.nan_to_num(self._data.data)
        self._data = self.tocsr()  # Converting back to 'csr'

    def round(self, decimals=0):
        """Evenly round to the given number of decimals."""
        x = self.tocsr()  # self.__class__ expects a csr
        data = np.round(x.data, decimals=decimals)
        # Round does not allocate new memory (data.flags.OWNDATA = False)
        # and indices/indptr must passed as copies
        return self.__class__(
            (data, x.indices, x.indptr), shape=self.shape, copy=True)

    def ceil(self):
        """Return the ceiling of the input, element-wise."""
        return self.__class__(self._data.ceil())

    def floor(self):
        """Return the floor of the input, element-wise."""
        return self.__class__(self._data.floor())

    def clip(self, c_min, c_max):
        """Clip (limit) the values in an array."""
        raise NotImplementedError

    def eliminate_zeros(self):
        self._data.eliminate_zeros()

    def sort(self, axis=-1, kind='quicksort', inplace=False):
        """Sort array."""
        if kind != 'quicksort':
            raise ValueError("only `quicksort` algorithm is supported")

        tosort = self if inplace is True else self.deepcopy()

        if axis == 1 or axis == -1:
            for i in range(tosort.shape[0]):
                row = tosort[i, :].todense()
                row.sort(axis=1, inplace=True)
                tosort[i, :] = row
        elif axis == 0:
            for i in range(tosort.shape[1]):
                column = tosort[:, i].todense()
                column.sort(axis=0, inplace=True)
                tosort[:, i] = column
        else:
            raise ValueError("wrong sorting axis.")

        return tosort

    def argsort(self, axis=-1, kind='quicksort'):
        """Returns the indices that would sort an array.

        If possible is better if you use sort function axis=-1 order based
        on last axis (which in sparse matrix is 1 horizontal).

        """
        if kind != 'quicksort':
            raise ValueError("only `quicksort` algorithm is supported")

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

        index_matrix = CDense().zeros(array.shape, dtype=int)

        axis_element = None
        for i in range(axis_elem_num):

            if axis == 1 or axis == -1 or axis is None:
                axis_element = array[i, :]  # order for row
            elif axis == 0:
                axis_element = array[:, i]  # order for column

            # argsort of current axis element
            sorted_data_idx = CDense(
                axis_element.todense()).argsort(axis=axis, kind='quicksort')

            if axis == 1 or axis == -1 or axis is None:
                index_matrix[i, :] = sorted_data_idx[0, :]  # order for row
            elif axis == 0:
                index_matrix[:, i] = sorted_data_idx[:, 0]  # order for column

        return index_matrix.ravel() if axis is None else index_matrix

    def shuffle(self):
        """Shuffle array data in-place."""
        if self.size > 0:
            if self.shape[0] > 1:  # only rows are shuffled
                shuffle_idx = CDense.randsample(self.shape[0], self.shape[0])
                self[:, :] = self[shuffle_idx, :]
            else:
                shuffle_idx = CDense.randsample(self.shape[1], self.shape[1])
                self[:, :] = self[0, shuffle_idx]

    # ------------ #
    # APPEND/MERGE #
    # ------------ #

    def append(self, array, axis=None):
        """Append an array along the given axis."""
        return self.__class__.concatenate(self, array, axis)

    def repmat(self, m, n):
        """Wrapper for repmat
        m: the number of times that we want repeat a alog axis 0 (vertical)
        n: the number of times that we want repeat a alog axis 1 (orizontal)
        """
        self_csr = self.tocsr()
        rows = [self_csr for _ in range(n)]
        blocks = [rows for _ in range(m)]
        if len(blocks) == 0:  # To manage the m = 0 case
            blocks = [[]]
        return self.__class__(scs.bmat(blocks, format='csr', dtype=self.dtype))

    def repeat(self, repeats, axis=None):
        """Repeat elements of an array."""
        raise NotImplementedError

    # ---------- #
    # COMPARISON #
    # ---------- #

    def logical_and(self, array):
        """Element-wise logical AND of array elements.

        Compare two arrays and returns a new array containing
        the element-wise logical AND.

        Parameters
        ----------
        array : CSparse
            The array like object holding the elements to compare
            current array with. Must have the same shape of first
            array.

        Returns
        -------
        CSparse
            The element-wise logical AND between the two arrays.

        """
        if self.shape != array.shape:
            raise ValueError(
                "array to compare must have shape {:}".format(self.shape))

        # This create an empty sparse matrix (basically full of zeros)
        and_result = self.__class__(self.shape, dtype=bool)

        # Ensure we have the expected type
        # Use 'coo' for fast conversion (if not a 'csr')
        x = self._tocoo_or_tocsr()
        x_array = array.tocoo() if \
            array._data.getformat() != 'csr' else array.tocsr()

        # Iterate over non-zero elements
        # This also works for any explicitly stored zero
        for e_i, e in enumerate(x.data):
            # Get indices of current element
            this_elem_row = self.nnz_indices[0][e_i]
            this_elem_col = self.nnz_indices[1][e_i]
            # Check if the 2nd array has an element in the same position
            y_same_bool = \
                (CDense(array.nnz_indices[0]) == this_elem_row).logical_and(
                    CDense(array.nnz_indices[1]) == this_elem_col)
            if y_same_bool.any():  # Found a corresponding element
                # Now extract the value to compare from second array
                same_position_val = int(x_array.data[y_same_bool.tondarray()])
                # Compare element from self with the one from 2nd array
                if np.logical_and(e, same_position_val):
                    and_result[this_elem_row, this_elem_col] = True

        return and_result

    def logical_or(self, array):
        """Element-wise logical OR of array elements.

        Compare two arrays and returns a new array containing
        the element-wise logical OR.

        Parameters
        ----------
        array : CSparse or array_like
            The array like object holding the elements to compare
            current array with. Must have the same shape of first
            array.

        Returns
        -------
        out_and : CSparse or bool
            The element-wise logical OR between the two arrays.

        """
        if self.shape != array.shape:
            raise ValueError(
                "array to compare must have shape {:}".format(self.shape))

        # All non-zero elements will be replaced with True, otherwise False
        out = self.astype(bool)

        # We now need to set as True the elements corresponding
        # to non-zeros in the 2nd array

        # Be sure there are only non-zeros in 2nd array
        array.eliminate_zeros()

        # Set as True any non-zero element of 2nd array
        out[array.nnz_indices] = True

        return out

    def logical_not(self):
        """Element-wise logical NOT of array elements."""
        # Be sure there are only non-zeros in array
        self.eliminate_zeros()
        # Create an array full of Trues (VERY expensive!)
        new_not = self.__class__(CDense.ones(self.shape)).astype(bool)
        # All old nonzeros should be zeros!
        new_not[self.nnz_indices] = False
        # Eliminate newly created zeros from internal csr_matrix structures
        new_not.eliminate_zeros()

        return new_not

    def maximum(self, array):
        """Element-wise maximum."""
        return self.__class__(
            self._data.maximum(self._buffer_to_builtin(array)))

    def minimum(self, array):
        """Element-wise minimum."""
        return self.__class__(
            self._data.minimum(self._buffer_to_builtin(array)))

    # ------ #
    # SEARCH #
    # ------ #

    def find(self, condition):
        """Indices of current array with True condition."""
        # size instead of shape as we just need one condition for each element
        if condition.size != self.size:
            raise ValueError("condition size must be {:}".format(self.size))
        # scs.find returns row indices, column indices, and nonzero data
        # we are interested only in row/column indices
        return list(map(list, scs.find(condition.tocsr())))[:2]

    def binary_search(self, value):
        raise NotImplementedError(
            "`binary_search` is not implemented for sparse arrays!")

    # ------------- #
    # DATA ANALYSIS #
    # ------------- #

    def get_nnz(self, axis=None):
        """Counts the number of non-zero values in the array.

        Parameters
        ----------
        axis : bool or None, optional
            Axis or tuple of axes along which to count non-zeros.
            Default is None, meaning that non-zeros will be counted
            along a flattened version of the array.

        Returns
        -------
        count : CDense or int
            Number of non-zero values in the array along a given axis.
            Otherwise, the total number of non-zero values in the
            array is returned.

        """
        # Result is dense (one element for each row/column)
        res = self.tocsr().getnnz(axis=axis)
        return CDense(res) if axis is not None else res

    def unique(self, return_index=False,
               return_inverse=False, return_counts=False):
        """Return unique array elements in dense format."""
        # Let's compute the number of zeros (will be used multiple times)
        n_zeros = self.size - self.nnz
        unique_items = [0] if n_zeros > 0 else []  # We have at least a zero?
        # Appending nonzero elements
        out = np.unique(self.tocsr().data,
                        return_index=return_index,
                        return_inverse=return_inverse,
                        return_counts=return_counts)
        if not any([return_index, return_inverse, return_counts]):
            # Return unique elements with correct dtype
            return CDense(unique_items + out.tolist()).astype(self.dtype)
        else:  # np.unique returned a tuple
            unique_items = CDense(
                unique_items + out[0].tolist()).astype(self.dtype)

        # If any extra parameter has been specified, output will be a tuple
        outputs = [unique_items]

        if return_index is True:

            # Indices will be extracted from flattened array
            flat_a = self.ravel()  # Returns a csr

            # csr indices must be sorted to extract unique indices
            if not bool(flat_a._data.has_sorted_indices):
                flat_a._data.sort_indices()

            # Let's get the index of the first zero...
            unique_index = CDense(dtype=int)
            if n_zeros > 0:  # ... if any!
                for i in range(flat_a.size):
                    # If a element is missing for indices[1]
                    # (nz column indices), means there is a zero there!
                    if i + 1 > len(flat_a.nnz_indices[1]) or \
                                    flat_a.nnz_indices[1][i] != i:
                        unique_index = CDense([i])
                        break

            # Let's get the indices of the nz elements (columns indices)
            unique_index = unique_index.append(
                CDense(flat_a.nnz_indices[1], dtype=int)[CDense(out[1])])
            # Add result to the list of returned items
            outputs.append(unique_index)

        if return_inverse is True:
            raise NotImplementedError(
                "`return_inverse` is currently not supported")

        if return_counts is True:
            # Let's check the number of extra parameters (to parse out)
            num_params = sum([return_index, return_inverse, return_counts])

            # Let's check the number of zeros
            counts_zeros = [n_zeros] if n_zeros > 0 else []

            # size of the out tuple depends on the number of extra params
            unique_counts = CDense(
                counts_zeros + out[min(3, num_params)].tolist(), dtype=int)
            # Add result to the list of returned items
            outputs.append(unique_counts)

        return tuple(outputs)

    def bincount(self, minlength=0):
        """Count the number of occurrences of each value in array 
        of non-negative ints."""
        # Use 'coo' for fast conversion (if not a 'csr')
        x = self._tocoo_or_tocsr()
        # Ensure we eliminate redundant zeros to obtain correct counts
        x.eliminate_zeros()
        # count number of elements (except zeros)
        nnz_bincount = np.bincount(x.data, minlength=minlength)
        # count number of zeros and set it
        nnz_bincount[0] = self.size - self.nnz
        return CDense(nnz_bincount)

    def norm(self, order=None):
        """Return the matrix norm of store data."""
        if is_int(order) and order < 0:
            # Scipy does not supports negative norms along axis
            raise NotImplementedError

        if self.size == 0:
            # Special handle as few norms raise error for empty arrays
            if order == 'fro':
                raise ValueError("Invalid norm order {:}.".format(order))
            return self.__class__([0.0])

        return CDense(norm(self.tocsr(), ord=order, axis=1)).astype(float)

    def norm_2d(self, order=None, axis=None, keepdims=True):
        """Return the matrix norm of store data."""
        if axis is not None and (is_int(order) and order < 0):
            # Scipy does not supports negative norms along axis
            raise NotImplementedError

        if axis is not None and order == 'fro':
            # 'fro' is a matrix norm
            raise ValueError("Invalid norm order {:}.".format(order))

        if self.size == 0:
            # Special handle as few norms raise error for empty arrays
            if axis is None and order in (2, -2):
                # Return an error consistent with scipy
                raise NotImplementedError
            if axis is None and order not in (
                    None, 'fro', inf, -inf, 1, -1):
                raise ValueError("Invalid norm order {:}.".format(order))
            return self.__class__([0.0])

        out = CDense(norm(self.tocsr(), ord=order, axis=axis)).astype(float)

        if axis is not None or keepdims is True:
            return out.atleast_2d().T if axis == 1 else out.atleast_2d()
        else:
            return out  # out is already a vector, so nothing to do

    def sum(self, axis=None, keepdims=True):
        """Sum of array elements over a given axis."""
        if self.size == 0:
            out_sum = CDense([[0.0]])
        else:
            out_sum = CDense(self._data.sum(axis))
        return \
            out_sum.ravel() if axis is None or keepdims is False else out_sum

    def cumsum(self, axis=None, dtype=None):
        """Return the cumulative sum of the array elements."""
        raise NotImplementedError

    def prod(self, axis=None, dtype=None, keepdims=True):
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
                    # Use 'coo' for fast conversion (if not a 'csr')
                    x = self._tocoo_or_tocsr()
                    return self.__class__(x.data.prod(), dtype=dtype)
                else:  # If any element is zero, product is zero
                    return self.__class__(0.0, dtype=dtype)
            elif axis == 0:
                out = CSparse((1, self.shape[1]), dtype=dtype)
                c_bincount = CDense(self.nnz_indices[1]).bincount()
                for e_idx, e in enumerate(c_bincount == self.shape[0]):
                    if bool(e) is True:
                        out[0, e_idx] = self[:, e_idx].todense().prod()
            elif axis == 1 or axis == -1:
                out = CSparse((self.shape[0], 1), dtype=dtype)
                c_bincount = CDense(self.nnz_indices[0]).bincount()
                for e_idx, e in enumerate(c_bincount == self.shape[1]):
                    if bool(e) is True:
                        out[e_idx, 0] = self[e_idx, :].todense().prod()
            else:
                raise ValueError("axis {:} is not valid".format(axis))

        return out.ravel() if axis is None or keepdims is False else out

    def all(self, axis=None, keepdims=True):
        """Return True if all array elements are boolean True."""
        if axis is not None or keepdims is not True:
            raise NotImplementedError(
                "`axis` and `keepdims` are currently not supported")
        # Use 'coo' for fast conversion (if not a 'csr')
        return bool(
            self.size == self.nnz and self._tocoo_or_tocsr().data.all())

    def any(self, axis=None, keepdims=True):
        """Return True if any array element is boolean True."""
        if axis is not None or keepdims is not True:
            raise NotImplementedError(
                "`axis` and `keepdims` are currently not supported")
        # Use 'coo' for fast conversion (if not a 'csr')
        return bool(self._tocoo_or_tocsr().data.any())

    def max(self, axis=None, keepdims=True):
        """Max of array elements over a given axis."""
        # Use 'coo' for fast conversion (if not a 'csr')
        out = self._tocoo_or_tocsr().max(axis=axis)
        if axis is None:  # return scalar
            return out
        out = CDense(out.toarray())
        return out.ravel() if keepdims is False else out

    def min(self, axis=None, keepdims=True):
        """Min of array elements over a given axis."""
        # Use 'coo' for fast conversion (if not a 'csr')
        out = self._tocoo_or_tocsr().min(axis=axis)
        if axis is None:  # return scalar
            return out
        out = CDense(out.toarray())
        return out.ravel() if keepdims is False else out

    def argmax(self, axis=None):
        """Indices of the maximum values along an axis.

        Parameters
        ----------
        axis : int, None, optional
            If None (default), array is flattened before computing
            index, otherwise the specified axis is used.

        Returns
        -------
        index : int, CDense
            Scalar with index of the maximum value for flattened array or
            CDense with indices along the given axis.

        Notes
        -----
        In case of multiple occurrences of the maximum values, the
        indices corresponding to the first occurrence are returned.

        """
        res = self._data.argmax(axis=axis)  # np.matrix or int
        return CDense(res) if axis is not None else res

    def argmin(self, axis=None):
        """Indices of the minimum values along an axis.

        Parameters
        ----------
        axis : int, None, optional
            If None (default), array is flattened before computing
            index, otherwise the specified axis is used.

        Returns
        -------
        index : int, CDense
            Scalar with index of the minimum value for flattened array or
            CDense with indices along the given axis.

        Notes
        -----
        In case of multiple occurrences of the minimum values, the
        indices corresponding to the first occurrence are returned.

        """
        res = self._data.argmin(axis=axis)  # np.matrix or int
        return CDense(res) if axis is not None else res

    def nanmax(self, axis=None, keepdims=True):
        raise NotImplementedError

    def nanmin(self, axis=None, keepdims=True):
        raise NotImplementedError

    def nanargmax(self, axis=None):
        raise NotImplementedError

    def nanargmin(self, axis=None):
        raise NotImplementedError

    def mean(self, axis=None, dtype=None, keepdims=True):
        """Mean of array elements over a given axis."""
        out = self._data.mean(axis=axis, dtype=dtype)
        if axis is None:  # return scalar
            return out
        return CDense(out).ravel() if keepdims is False else CDense(out)

    def median(self, axis=None, keepdims=True):
        """Median of array elements over a given axis."""
        raise NotImplementedError

    def std(self, axis=None, ddof=0, keepdims=True):
        """Standard deviation of matrix over the given axis."""
        array_mean = CDense(self.mean(axis=axis)).atleast_2d()

        centered_array = self - array_mean.repmat(
            [1 if array_mean.shape[0] == self.shape[0] else self.shape[0]][0],
            [1 if array_mean.shape[1] == self.shape[1] else self.shape[1]][0])
        # n is array size for axis == None or
        # the number of rows/columns of specified axis
        n = self.size if axis is None else self.shape[axis]
        variance = (1.0 / (n - ddof)) * (centered_array ** 2)

        return CDense(variance.sum(axis=axis, keepdims=keepdims).sqrt())

    def sha1(self):
        """Calculate the sha1 hexadecimal hash of array.

        Returns
        -------
        hash : str
            Hexadecimal hash of array.

        """
        import hashlib
        x = self.tocsr()

        h = hashlib.new('sha1')

        # Hash by taking into account shape and sparse matrix internals
        h.update(hex(hash(x.shape)).encode('utf-8'))
        # The returned sha1 could be different for same data
        # but different memory order. Use C order to be consistent
        h.update(np.ascontiguousarray(x.indices))
        h.update(np.ascontiguousarray(x.indptr))
        h.update(np.ascontiguousarray(x.data))

        return h.hexdigest()

    def is_inf(self):
        """Test element-wise for positive or negative infinity."""
        # Get the indices of array where inf values are
        return _expand_nnz_bool(self, self.nnz_data.is_inf())

    def is_posinf(self):
        """Test element-wise for positive infinity."""
        # Get the indices of array where +inf values are
        return _expand_nnz_bool(self, self.nnz_data.is_posinf())

    def is_neginf(self):
        """Test element-wise for negative infinity."""
        # Get the indices of array where -inf values are
        return _expand_nnz_bool(self, self.nnz_data.is_neginf())

    def is_nan(self):
        """Test element-wise for Not a Number (NaN)."""
        # Get the indices of array where nan values are
        return _expand_nnz_bool(self, self.nnz_data.is_nan())

    # ----------------- #
    # MATH ELEMENT-WISE #
    # ----------------- #

    def sqrt(self):
        """Return the element-wise square root of array."""
        return self.__class__(self._data.sqrt())

    def sin(self):
        """Trigonometric sine, element-wise."""
        return self.__class__(self.tocsr().sin())

    def cos(self):
        """Trigonometric cosine, element-wise."""
        raise NotImplementedError("`cos` is not available for sparse arrays!")

    def exp(self):
        """Exponential, element-wise."""
        raise NotImplementedError("`exp` is not available for sparse arrays!")

    def log(self):
        """Natural logarithm, element-wise."""
        raise NotImplementedError("`exp` is not available for sparse arrays!")

    def log10(self):
        """Base 10 logarithm, element-wise."""
        raise NotImplementedError(
            "`log10` is not available for sparse arrays!")

    def pow(self, exp):
        """Array elements raised to powers from input exponent, element-wise.

        Equivalent to standard ``**`` operator.

        Parameters
        ----------
        exp : scalar
            Exponent of power, single scalar.

        Returns
        -------
        pow_array : CSparse
            New array with the power of current data using
            input exponents.

        """
        return self.__pow__(exp)

    def normpdf(self, mu=0.0, sigma=1.0):
        """Return normal distribution function."""
        raise NotImplementedError(
            "`normpdf` is not available for sparse arrays!")

    # ----- #
    # MIXED #
    # ----- #

    def sign(self):
        """Return the element-wise sign of array."""
        return self.__class__(self._data.sign())

    def diag(self, k=0):
        """Extract a diagonal or construct a diagonal array."""
        if self.shape[0] == 1:
            return self.__class__(scs.diags(
                self.tondarray(), offsets=[k], format='csr', dtype=self.dtype))
        else:
            if (k > 0 and k > self.shape[1] - 1) or \
                    (k < 0 and abs(k) > self.shape[0] - 1):
                raise ValueError("k exceeds matrix dimensions")
            return CDense(self.tocsr().diagonal(k=k))

    def dot(self, array):
        # Only matrix multiplication is supported for sparse arrays
        if len(array.shape) == 1:  # We work with 2D arrays
            array = array.atleast_2d()
        return self.__class__(self._data.dot(self._buffer_to_builtin(array)))

    def interp(self, x_data, y_data, return_left=None, return_right=None):
        """One-dimensional linear interpolation."""
        raise NotImplementedError(
            "`interp` is not available for sparse arrays!")

    def inv(self):
        """Compute the (multiplicative) inverse of a square matrix."""
        # scipy.sparse.linalg.spsolve is more efficient on csc arrays
        return self.__class__(inv(self.tocsc()))

    def pinv(self, rcond=1e-15):
        """Compute the (Moore-Penrose) pseudo-inverse of a matrix."""
        raise NotImplementedError

    # -------------------------------- #
    # # # # # # CLASSMETHODS # # # # # #
    # ---------------------------------#

    @classmethod
    def empty(cls, shape, dtype=float):
        """Return a new array of given shape and type, without filling it."""
        return cls(scs.csr_matrix(shape, dtype=dtype))

    @classmethod
    def zeros(cls, shape, dtype=float):
        """Return a new array of given shape and type, without filling it."""
        return cls(scs.csr_matrix(shape, dtype=dtype))

    @classmethod
    def ones(cls, shape, dtype=float):
        """Return a new array of given shape and type, filled with ones."""
        raise NotImplementedError

    @classmethod
    def eye(cls, n_rows, n_cols=None, k=0, dtype=float):
        """Return an array of desired dimension with ones on the diagonal
        and zeros elsewhere.

        See scipy.sparse.eye for more information.

        Parameters
        ----------
        n_rows : number of rows for output array, integer.
        n_cols : number of columns in the output. If None, defaults to n_rows.
        k : index of the diagonal. 0 (the default) refers to the main diagonal,
            a positive value refers to an upper diagonal, and a negative value
            to a lower diagonal.
        dtype : datatype of array data.

        Returns
        -------
        Sparse array of desired shape with ones on the diagonal and
        zeros elsewhere.

        """
        return cls(scs.eye(n_rows, n_cols, k=k, dtype=dtype, format='csr'))

    @classmethod
    def rand(cls, shape, random_state=None, density=0.01):
        """Wrapper for scipy.sparse.rand.

        Creates a random sparse array of [0, 1] floats
        with input density and shape.
        Density equal to one means a dense matrix,
        density of 0 means a matrix with no non-zero items.

        """
        n_rows, n_cols = shape  # Unpacking the shape
        return cls(scs.rand(n_rows, n_cols, density=density, format='csr'))

    @classmethod
    def randn(cls, shape, random_state=None):
        raise NotImplementedError

    @classmethod
    def randuniform(cls, low=0.0, high=1.0, shape=None, random_state=None):
        """Return random samples from low (inclusive) to high (exclusive)."""
        raise NotImplementedError

    @classmethod
    def randint(cls, low, high=None, shape=None, random_state=None):
        """Return random integers from low (inclusive) to high (exclusive)."""
        raise NotImplementedError

    @classmethod
    def randsample(cls, a, shape=None, replace=False, random_state=None):
        """Generates a random sample from a given array."""
        raise NotImplementedError

    @classmethod
    def linspace(cls, start, stop, num=50, endpoint=True):
        """Return evenly spaced numbers over a specified interval."""
        raise NotImplementedError

    @classmethod
    def arange(cls, start=None, stop=None, step=1, dtype=None):
        """Return evenly spaced values within a given interval."""
        raise NotImplementedError

    @classmethod
    def concatenate(cls, array1, array2, axis=1):
        """Concatenate a sequence of arrays along the given axis."""
        if not isinstance(array1, cls) or not isinstance(array2, cls):
            raise TypeError(
                "both arrays to concatenate must be {:}".format(cls))

        if axis is None:  # both arrays should be ravelled
            array1 = array1.ravel()
            array2 = array2.ravel()
            axis = 1  # Simulate an horizontal concatenation

        if axis == 0:  # Vertical
            return cls(scs.vstack([array1.tocsr(), array2.tocsr()]))
        elif axis == 1:  # Horizontal
            return cls(scs.hstack([array1.tocsc(), array2.tocsc()]))
        else:
            raise ValueError("axis should be one of {0, 1, None}")

    @classmethod
    def comblist(cls, list_of_list, dtype=float):
        """Return the norm of store data."""
        raise NotImplementedError

    @classmethod
    def meshgrid(cls, xi, indexing='xy'):
        """Return coordinate matrices from coordinate vectors."""
        raise NotImplementedError
