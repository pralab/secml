"""
.. module:: ArrayInterface
   :synopsis: Class that defines an interface for array classes

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod, abstractproperty
from copy import deepcopy


class _CArrayInterface(object):

    __metaclass__ = ABCMeta

    # ------------------------------ #
    # # # # # # PROPERTIES # # # # # #
    # -------------------------------#

    @abstractproperty
    def shape(self):
        """Shape of stored data, tuple of ints.

        Returns
        -------
        tuple of int
            Returns the shape of the array. Tuple with one integer
            for each array's dimension.

        """
        raise NotImplementedError

    @abstractproperty
    def size(self):
        """Size (number of elements) of array.

        For sparse data, this counts both zeros and non-zero elements.

        Returns
        -------
        int
            Returns the size of the array.

        """
        raise NotImplementedError

    @abstractproperty
    def ndim(self):
        """Number of array dimensions.

        This is always 2 for sparse arrays.

        Returns
        -------
        int
            Returns the number of dimensions of the array.

        """
        raise NotImplementedError

    @abstractproperty
    def dtype(self):
        """Data-type of stored data.

        Returns
        -------
        dtype
            Returns the data type of stored data.

        """
        raise NotImplementedError

    @abstractproperty
    def nnz(self):
        """Number of non-zero array elements.

        Returns
        -------
        int
            Number of non-zero array elements.

        """
        raise NotImplementedError

    @abstractproperty
    def nnz_indices(self):
        """Index of non-zero array elements.

        Returns
        -------
        list
            List of 2 lists. Inside out[0] there are
            the indices of the corresponding rows and inside out[1]
            there are the indices of the corresponding columns of
            non-zero array elements.

        """
        raise NotImplementedError

    @abstractproperty
    def nnz_data(self):
        """Return non-zero array elements.

        Returns
        -------
        CArrayInterface
            Flat array, dense, shape (n, ), with non-zero array elements.

        """
        raise NotImplementedError

    @abstractproperty
    def T(self):
        """Transposed array data.

        Returns
        -------
        CArrayInterface
            Transposed array.

        """
        raise NotImplementedError

    # --------------------------- #
    # # # # # # CASTING # # # # # #
    # ----------------------------#

    @abstractmethod
    def tondarray(self):
        """Return a dense numpy.ndarray representation of array.

        Returns
        -------
        numpy.ndarray
            A representation of current data as numpy.ndarray.
            If possible, we avoid copying original data.

        """
        raise NotImplementedError

    @abstractmethod
    def tocsr(self):
        """Return a sparse scipy.sparse.csr_matrix representation of array.

        Returns
        -------
        scipy.sparse.csr_matrix
            A representation of current data as scipy.sparse.csr_matrix.
            If possible, we avoid copying original data.

        """
        raise NotImplementedError

    @abstractmethod
    def tolist(self):
        """Return the array as a (possibly nested) list.

        Returns
        -------
        list
            The possibly nested list of array elements.

        """
        raise NotImplementedError

    # ---------------------------- #
    # # # # # # INDEXING # # # # # #
    # -----------------------------#

    @abstractmethod
    def __getitem__(self, idx):
        """Return a new array with slicing/indexing result."""
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, idx, value):
        """Set input data to slicing/indexing result."""
        raise NotImplementedError

    # ------------------------------------ #
    # # # # # # SYSTEM OVERLOADS # # # # # #
    # -------------------------------------#

    @abstractmethod
    def __add__(self, other):
        """Element-wise addition.

        Parameters
        ----------
        other : CArrayInterface or scalar or bool
            Element to add to current array.
            If a CArrayInterface, element-wise addition will be performed.
            If scalar or boolean, the element will be sum
            to each array element.

        Returns
        -------
        CArrayInterface
            Array after addition.

        """
        raise NotImplementedError

    @abstractmethod
    def __radd__(self, other):
        """Element-wise (inverse) addition.

        Parameters
        ----------
        other : scalar or bool
            Element to add to current array.
            The element will be sum to each array element.

        Returns
        -------
        CArrayInterface
            Array after addition. Format is preserved.

        """
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other):
        """Element-wise subtraction.

        Parameters
        ----------
        other : CArrayInterface or scalar or bool
            Element to subtract to current array.
            If a CArrayInterface, element-wis subtraction will be performed.
            If scalar or boolean, the element will be subtracted
            to each array element.

        Returns
        -------
        CArrayInterface
            Array after subtraction.

        """
        raise NotImplementedError

    @abstractmethod
    def __rsub__(self, other):
        """Element-wise (inverse) subtraction.

        Parameters
        ----------
        other : scalar or bool
            Element to subtract to current array.
            The element will be subtracted to each array element.

        Returns
        -------
        CArrayInterface
            Array after subtraction.

        """
        raise NotImplementedError

    @abstractmethod
    def __mul__(self, other):
        """Element-wise product.

        Parameters
        ----------
        other : CArrayInterface or scalar or bool
            Element to multiply to current array.
            If a CArrayInterface, element-wise product will be performed.
            If scalar or boolean, the element will be multiplied
            to each array element.

        Returns
        -------
        CArrayInterface
            Array after product.

        """
        raise NotImplementedError

    @abstractmethod
    def __rmul__(self, other):
        """Element-wise (inverse) product.

        Parameters
        ----------
        other : scalar or bool
            Element to multiply to current array.
            The element will be multiplied to each array element.

        Returns
        -------
        CArrayInterface
            Array after product.

        """
        raise NotImplementedError

    @abstractmethod
    def __truediv__(self, other):
        """Element-wise true division.

        Parameters
        ----------
        other : CArrayInterface or scalar or bool
            Element to divide to current array.
            If a CArrayInterface, element-wise division will be performed.
            If scalar or boolean, the element will be divided
            to each array element.

        Returns
        -------
        CArrayInterface
            Array after division.

        """
        raise NotImplementedError

    @abstractmethod
    def __rtruediv__(self, other):
        """Element-wise (inverse) true division.

        Parameters
        ----------
        other : scalar or bool
            Element to divide to current array.
            The element will be divided to each array element.

        Returns
        -------
        CArrayInterface
            Array after true division.

        """
        raise NotImplementedError

    @abstractmethod
    def __div__(self, other):
        """Element-wise division. True division will be performed.

        See __truediv__() for more informations.

        """
        raise NotImplementedError

    @abstractmethod
    def __rdiv__(self, other):
        """Element-wise (inverse) division. True division will be performed.

        See .__rtruediv__() for more informations.

        """
        raise NotImplementedError

    @abstractmethod
    def __floordiv__(self, other):
        """Element-wise floor division (// operator).

        Parameters
        ----------
        other : CArrayInterface or scalar or bool
            Element to divide to current array.
            If a CArrayInterface, element-wise division will be performed.
            If scalar or boolean, the element will be divided
            to each array element.

        Returns
        -------
        CArrayInterface
            Array after floor division.

        """
        raise NotImplementedError

    @abstractmethod
    def __rfloordiv__(self, other):
        """Element-wise (inverse) floor division (// operator).

        Parameters
        ----------
        other : scalar or bool
            Element to divide to current array.
            The element will be divided to each array element.

        Returns
        -------
        CArrayInterface
            Array after division. Array format is always preserved.

        """
        raise NotImplementedError

    @abstractmethod
    def __abs__(self):
        """Returns array elements without sign.

        Returns
        -------
        CArrayInterface
            Array with the corresponding elements without sign.

        """
        raise NotImplementedError

    @abstractmethod
    def __neg__(self):
        """Returns array elements with negated sign.

        Returns
        -------
        CArrayInterface
            Array with the corresponding elements with negated sign.

        """
        raise NotImplementedError

    @abstractmethod
    def __pow__(self, power):
        """Element-wise power.

        Parameters
        ----------
        power : CArrayInterface or scalar or bool
            Power to use. If scalar or boolean, each array element will be
            elevated to power. If a CArrayInterface, each array element
            will be elevated to the corresponding element of the input array.

        Returns
        -------
        CArrayInterface
            Array after power. Array format is always preserved.

        """
        raise NotImplementedError

    @abstractmethod
    def __rpow__(self, power):
        """Element-wise (inverse) power.

        Parameters
        ----------
        power : scalar or bool
            Power to use. Each array element will be elevated to power.

        Returns
        -------
        CArrayInterface
            Array after power. Array format is always preserved.

        """
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        """Element-wise == operator.

        Parameters
        ----------
        other : CArrayInterface or scalar or bool
            Element to be compared.
            If a CArrayInterface, element-wise comparison will be performed.
            If scalar or boolean, the element will be compared
            to each array element.

        Returns
        -------
        CArrayInterface
            Boolean array with comparison result.

        """
        raise NotImplementedError

    @abstractmethod
    def __lt__(self, other):
        """Element-wise < operator.

        Parameters
        ----------
        other : CArrayInterface or scalar or bool
            Element to be compared.
            If a CArray, element-wise comparison will be performed.
            If scalar or boolean, the element will be compared
            to each array element.

        Returns
        -------
        CArrayInterface
            Boolean array with comparison result.

        """
        raise NotImplementedError

    @abstractmethod
    def __le__(self, other):
        """Element-wise <= operator.

        Parameters
        ----------
        other : CArrayInterface or scalar or bool
            Element to be compared.
            If a CArrayInterface, element-wise comparison will be performed.
            If scalar or boolean, the element will be compared
            to each array element.

        Returns
        -------
        CArrayInterface
            Boolean array with comparison result.

        """
        raise NotImplementedError

    @abstractmethod
    def __gt__(self, other):
        """Element-wise > operator.

        Parameters
        ----------
        other : CArrayInterface or scalar or bool
            Element to be compared.
            If a CArray, element-wise comparison will be performed.
            If scalar or boolean, the element will be compared
            to each array element.

        Returns
        -------
        CArrayInterface
            Boolean array with comparison result.

        """
        raise NotImplementedError

    @abstractmethod
    def __ge__(self, other):
        """Element-wise >= operator.

        Parameters
        ----------
        other : CArrayInterface or scalar or bool
            Element to be compared.
            If a CArrayInterface, element-wise comparison will be performed.
            If scalar or boolean, the element will be compared
            to each array element.

        Returns
        -------
        CArrayInterface
            Boolean array with comparison result.

        """
        raise NotImplementedError

    @abstractmethod
    def __ne__(self, other):
        """Element-wise != operator.

        Parameters
        ----------
        other : CArrayInterface or scalar or bool
            Element to be compared. If a CArray, element-wise
            comparison will be performed. If scalar or boolean,
            the element will be compared to each array element.

        Returns
        -------
        CArrayInterface
            Boolean array with comparison result.

        """
        raise NotImplementedError

    @abstractmethod
    def __bool__(self):
        """Manage 'and' and 'or' operators."""
        raise NotImplementedError

    __nonzero__ = __bool__  # Compatibility with python < 3

    @abstractmethod
    def __iter__(self):
        """Yields array elements in raster-scan order.

        Yields
        ------
        scalar
            Each array's element in raster-scan order.

        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        """Define `print` (or `str`) behaviour."""
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        """Define `repr` behaviour of array."""
        raise NotImplementedError

    # ------------------------------ #
    # # # # # # COPY UTILS # # # # # #
    # -------------------------------#

    @abstractmethod
    def __copy__(self):
        """Called when copy.copy is called."""
        raise NotImplementedError

    def deepcopy(self):
        """Return a deepcopy of current array."""
        return deepcopy(self)

    @abstractmethod
    def __deepcopy__(self, memo):
        """Called when copy.deepcopy is called."""
        raise NotImplementedError

    # ----------------------------- #
    # # # # # # SAVE/LOAD # # # # # #
    # ------------------------------#

    @abstractmethod
    def save(self, datafile, overwrite=False):
        """Save array data into plain text file.

        Data is stored preserving original data type.

        Parameters
        ----------
        datafile : str or file_handle
            Text file to save data to. If a string, it's supposed
            to be the filename of file to save. If a file handle,
            data will be stored using active file handle mode.
            If the filename ends in .gz, the file is automatically
            saved in compressed gzip format. load() function understands
            gzipped files transparently.
        overwrite : bool, optional
            If True and target file already exists, file will be overwritten.
            Otherwise (default), IOError will be raised.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, datafile, dtype=float):
        """Load array data from plain text file.

        Parameters
        ----------
        datafile : str or file_handle
            File or filename to read. If the filename extension
            is gz or bz2, the file is first decompressed.
        dtype : str or dtype, optional
            Data type of the resulting array, default 'float'.
            If None, the dtype will be determined by the contents of the file.

        Returns
        -------
        CArrayInterface
            Array resulting from loading.

        """
        raise NotImplementedError

    # ----------------------------- #
    # # # # # # UTILITIES # # # # # #
    # ------------------------------#

    # ---------------- #
    # SHAPE ALTERATION #
    # ---------------- #

    @abstractmethod
    def transpose(self):
        """Returns current array with axes transposed.

        A view is returned if possible.

        Returns
        -------
        CArrayInterface
            A view, if possible, of current array with axes suitably permuted.

        """
        raise NotImplementedError

    @abstractmethod
    def ravel(self):
        """Return a flattened array.

        A copy is made only if needed.

        Returns
        -------
        CArrayInterface
            Flattened view (if possible) of the array.

        """
        raise NotImplementedError

    @abstractmethod
    def flatten(self):
        """Return a flattened copy of array.

        Returns
        -------
        CArrayInterface
            Flattened copy of the array.

        """
        raise NotImplementedError

    @abstractmethod
    def atleast_2d(self):
        """Return the array with at least two dimensions.

        A copy is made only if needed.

        Returns
        -------
        CArrayInterface
            Array with ndim >= 2.

        """
        raise NotImplementedError

    @abstractmethod
    def reshape(self, shape):
        """Gives a new shape to an array without changing its data.

        Parameters
        ----------
        shape : int or sequence of ints
            Desired shape for output array. One integer for each axis.

        Returns
        -------
        CArrayInterface
            Array with new shape. If possible, a view of original array data
            will be returned, otherwise a copy will be made first.

        """
        raise NotImplementedError

    @abstractmethod
    def resize(self, shape, constant=0):
        """Return a new array with the specified shape.

        Missing entries are filled with input constant (default 0).

        Parameters
        ----------
        shape : int or sequence of ints
            Integer or one integer for each desired dimension of output array.
        constant : scalar
            Scalar to be used for filling missing entries. Default 0.

        Returns
        -------
        out : CArray
            Array with new shape. Array dtype is preserved.
            Missing entries are filled with the desired constant (default 0).

        """
        raise NotImplementedError

    # --------------- #
    # DATA ALTERATION #
    # --------------- #

    @abstractmethod
    def astype(self, dtype):
        """Copy of the array, casted to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.

        Returns
        -------
        CArrayInterface
            Copy of the original array casted to new data type.

        """
        raise NotImplementedError

    @abstractmethod
    def nan_to_num(self):
        """Replace nan with zero and inf with finite numbers.

        Replace array elements if Not a Number (NaN) with zero,
        if (positive or negative) infinity with the largest
        (smallest or most negative) floating point value that
        fits in the array dtype. All finite numbers are upcast
        to the output dtype (default float64).

        Notes
        -----
        We use the IEEE Standard for Binary Floating-Point for
        Arithmetic (IEEE 754). This means that Not a Number
        is not equivalent to infinity.

        """
        raise NotImplementedError

    @abstractmethod
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
        CArrayInterface
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

        """
        raise NotImplementedError

    @abstractmethod
    def ceil(self):
        """Return the ceiling of the input, element-wise.

        The ceil of the scalar x is the smallest integer i, such that i >= x.

        Returns
        -------
        CArrayInterface
            The ceiling of each element in x, with float dtype.

        """
        raise NotImplementedError

    @abstractmethod
    def floor(self):
        """Return the floor of the input, element-wise.

        The floor of the scalar x is the largest integer i, such that i <= x.

        Returns
        -------
        CArrayInterface
            The floor of each element in x, with float dtype.

        Notes
        -----
        Some spreadsheet programs calculate the "floor-towards-zero",
        in other words floor(-2.5) == -2. We instead uses the
        definition of floor where floor(-2.5) == -3.

        """
        raise NotImplementedError

    @abstractmethod
    def clip(self, c_min, c_max):
        """Clip (limit) the values in an array.

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
        CArrayInterface
            Returns a new array containing the clipped array elements.
            Dtype of the output array depends on the dtype of original array
            and on the dtype of the clipping limits.

        """
        raise NotImplementedError

    def abs(self):
        """Returns array elements without sign.

        Returns
        -------
        CArrayInterface
            Array with the corresponding elements without sign.

        """
        return abs(self)

    @abstractmethod
    def sort(self, axis=-1, kind='quicksort', inplace=False):
        """Sort an array.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. The default is -1 (the last axis).
        kind : {'quicksort', 'mergesort', 'heapsort'}, optional, dense only
            Sorting algorithm to use. Default 'quicksort'.
        inplace : bool, optional
            If True, array will be sorted in-place. Default False.

        Returns
        -------
        CArrayInterface
            Sorted array. This will be a new array only if inplace is False.

        """
        raise NotImplementedError

    @abstractmethod
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
        kind : {'quicksort', 'mergesort', 'heapsort'}, optional
            Sorting algorithm to use. Default 'quicksort'.
            For sparse arrays, only 'quicksort' is available.

        Returns
        -------
        CArrayInterface
            Array of indices that sort the array along the specified axis.
            In other words, array[index_array] yields a sorted array.

        """
        raise NotImplementedError

    @abstractmethod
    def shuffle(self):
        """Modify array in-place by shuffling its contents.

        This function only shuffles the array along the first
        index of a not vector-like, multi-dimensional array.

        """
        raise NotImplementedError

    # ------------ #
    # APPEND/MERGE #
    # ------------ #

    @abstractmethod
    def append(self, array, axis=None):
        """Append values to the end of an array.

        Parameters
        ----------
        array : CArrayInterface or array_like
            Second array.
        axis : int or None, optional
            The axis along which values are appended.
            If axis is None, both arrays are flattened before use.

        Returns
        -------
        CArrayInterface
            A copy of array with values appended to axis. Note that append
            does not occur in-place: a new array is allocated and filled.
            If axis is None, out is a flattened array.

        """
        raise NotImplementedError

    @abstractmethod
    def repmat(self, m, n):
        """Repeat an array M x N times.

        Parameters
        ----------
        m, n : int
            The number of times the array is repeated along
            the first and second axes.

        Returns
        -------
        CArrayInterface
            The result of repeating array m X n times.

        """
        raise NotImplementedError

    @abstractmethod
    def repeat(self, repeats, axis=None):
        """Repeat elements of an array.

        DENSE FORMAT ONLY

        Parameters
        ----------
        repeats : int or list or CArrayInterface
            The number of repetitions for each element. If this is
            a list or a CArrayInterface object, it will be broadcasted
            to fit the shape of the given axis.
        axis : int, optional
            The axis along which to repeat values. By default, array
            is flattened before use.

        Returns
        -------
        CArrayInterface
            Output array which has the same shape as original array,
            except along the given axis. If axis is None, a flat array
            is returned.

        """
        raise NotImplementedError

    # ---------- #
    # COMPARISON #
    # ---------- #

    @abstractmethod
    def logical_and(self, array):
        """Element-wise logical AND of array elements.

        Compare two arrays and returns a new array containing
        the element-wise logical AND.

        Parameters
        ----------
        array : CArrayInterface
            The array like object holding the elements to compare
            current array with. Must have the same shape of first array.

        Returns
        -------
        CArrayInterface
            The element-wise logical AND between the two arrays.

        """
        raise NotImplementedError

    @abstractmethod
    def logical_or(self, array):
        """Element-wise logical OR of array elements.

        Compare two arrays and returns a new array containing
        the element-wise logical OR.

        Parameters
        ----------
        array : CArrayInterface
            The array like object holding the elements to compare
            current array with. Must have the same shape of first array.

        Returns
        -------
        CArrayInterface
            The element-wise logical OR between the two arrays.

        """
        raise NotImplementedError

    @abstractmethod
    def logical_not(self):
        """Element-wise logical NOT of array elements.

        Returns
        -------
        CArrayInterface
            The element-wise logical NOT.

        """
        raise NotImplementedError

    @abstractmethod
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
        array : CArrayInterface
            The array like object holding the elements to compare
            current array with. Must have the same shape of first array.

        Returns
        -------
        CArrayInterface
            The element-wise maximum between the two arrays.

        """
        raise NotImplementedError

    @abstractmethod
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
        array : CArray
            The array like object holding the elements to compare
            current array with. Must have the same shape of first array.

        Returns
        -------
        CArray
            The element-wise minimum between the two arrays.

        """
        raise NotImplementedError

    # ------ #
    # SEARCH #
    # ------ #

    @abstractmethod
    def find(self, condition):
        """Returns array elements indices depending on condition.

        Parameters
        ----------
        condition : CArrayInterface
            Array with booleans representing desired condition.

        Returns
        -------
        list
            List of len(out_find) == ndim with indices corresponding to
            array elements where condition is True. Es. for matrices,
            out_find[0] holds the indices of rows, out_find[1] the
            indices of columns.

        """
        raise NotImplementedError

    @abstractmethod
    def binary_search(self, value):
        """Returns the index of each input value inside the array.

        If value is not found inside the array, the index
        of the closest value will be returned.
        Array will be flattened before search.

        Parameters
        ----------
        value : scalar or CArrayInterface
            Element or array of elements to search inside
            the flattened array.

        Returns
        -------
        int or CArrayInterface
            Position of input value, or the closest one, inside
            flattened array. If `value` is an array, a CArrayInterface
            with the position of each `value` element is returned.

        """
        raise NotImplementedError

    # ------------- #
    # DATA ANALYSIS #
    # ------------- #

    @abstractmethod
    def unique(self, return_index=False,
               return_inverse=False, return_counts=False):
        """Find the unique elements of an array.

        There are three optional outputs in addition to the unique elements:
         - the indices of the input array that give the unique values
         - the indices of the unique array that reconstruct the input array
         - the number of times each unique value comes up in the input array

        Parameters
        ----------
        return_index : bool, optional
            If True, also return the indices of array that result
            in the unique array (default False).
        return_inverse : bool, optional
            If True, also return the indices of the unique array
            that can be used to reconstruct the original array
            (default False).
        return_counts : bool, optional
            If True, also return the number of times each unique item appears.

        Returns
        -------
        unique : CArrayInterface
            Dense array with the sorted unique values of the array.
        unique_index : CArrayInterface, optional
            The indices of the first occurrences of the unique values
            in the (flattened) original array. Only provided if
            return_index is True.
        unique_inverse : CArrayInterface, optional
            The indices to reconstruct the (flattened) original array
            from the unique array. Only provided if return_inverse is True.
        unique_counts : CArrayInterface, optional
            The number of times each unique item appears in the original array.
            Only provided if return_counts is True.

        """
        raise NotImplementedError

    @abstractmethod
    def bincount(self):
        """Count number of occurrences of each value in array of non-negative ints.

        Only flat arrays of integer dtype are supported.

        Returns
        -------
        CArrayInterface
            The occurrence number for every different element of array.
            The length of output array is equal to a.max()+1.

        """
        raise NotImplementedError

    @abstractmethod
    def norm(self, order=None):
        """Array norm.

        The supported norm types depend on the array number of dimensions.

        Parameters
        ----------
        order : {'fro', int, np.inf, -np.inf}, optional
            Order of the norm (see table under Notes).
        axis : int or None, optional
            If axis is an integer, it specifies the axis of array along
            which to compute the norms. If axis is None then
            the norm of the entire array is computed.
        keepdims : bool, optional
            If this is set to True (default), the result will
            broadcast correctly against the original array.
            Otherwise resulting array is flattened.

        Returns
        -------
        float or CArrayInterface
            Norm of the array.

        """
        raise NotImplementedError

    @abstractmethod
    def norm_2d(self, order=None, axis=None, keepdims=True):
        """Array norm.

        The supported norm types depend on the array number of dimensions.

        Parameters
        ----------
        order : {'fro', int, np.inf, -np.inf}, optional
            Order of the norm (see table under Notes).
        axis : int or None, optional
            If axis is an integer, it specifies the axis of array along
            which to compute the norms. If axis is None then
            the norm of the entire array is computed.
        keepdims : bool, optional
            If this is set to True (default), the result will
            broadcast correctly against the original array.
            Otherwise resulting array is flattened.

        Returns
        -------
        float or CArrayInterface
            Norm of the array.

        """
        raise NotImplementedError

    @abstractmethod
    def sum(self, axis=None, keepdims=True):
        """Sum of array elements over a given axis.

        Parameters
        ----------
        axis : int or None, optional
            Axis along which a sum is performed. The default
            (axis = None) is perform a sum over all the
            dimensions of the input array. axis may be negative,
            in which case it counts from the last to the first axis.
        keepdims : bool, optional
            If this is set to True (default), the result will
            broadcast correctly against the original array.
            Otherwise resulting array is flattened.

        Returns
        -------
        scalar or CArrayInterface
            A 2-dim array with the elements sum along specified axis.
            If axis is None, a scalar is returned.

        """
        raise NotImplementedError

    @abstractmethod
    def cumsum(self, axis=None):
        """Return the cumulative sum of the array elements along a given axis.

        DENSE FORMAT ONLY

        Parameters
        ----------
        axis : int or None, optional
            Axis along which the cumulative sum is computed.
            The default (None) is to compute the cumsum over
            the flattened array.

        Returns
        -------
        CArrayInterface
            New array with cumulative sum of elements.
            If axis is None, flat array with same size of input array.
            If axis is not None, same shape of input array.

        """
        raise NotImplementedError

    @abstractmethod
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
        scalar or CArrayInterface
            A 2-dim array with the elements product along specified axis.
            If axis is None, a scalar is returned.

        """
        raise NotImplementedError

    @abstractmethod
    def all(self, axis=None, keepdims=True):
        """Test whether all array elements along a given axis evaluate to True.

        Parameters
        ----------
        axis : int or None, optional
            Axis or axes along which logical AND between
            elements is performed. The default (axis = None)
            is to perform a logical AND over all the dimensions
            of the input array. If axis is negative, it counts
            from the last to the first axis.
        keepdims : bool, optional
            If this is set to True (default), the result will
            broadcast correctly against the original array.
            Otherwise resulting array is flattened.

        Returns
        -------
        bool or CArrayInterface
            A new boolean or array with logical AND element-wise.

        """
        raise NotImplementedError

    @abstractmethod
    def any(self, axis=None, keepdims=True):
        """Test whether any array elements along a given axis evaluate to True.

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
        bool or CArrayInterface
            A new boolean or array with logical OR element-wise.

        """
        raise NotImplementedError

    @abstractmethod
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
        scalar or CArrayInterface
            Maximum of array.
            If axis is None, the result is a scalar value.
            If axis is given, the result will
             broadcast correctly against the original array.

        """
        raise NotImplementedError

    @abstractmethod
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
        scalar or CArrayInterface
            Minimum of array.
            If axis is None, the result is a scalar value.
            If axis is given, the result will
             broadcast correctly against the original array.

        """
        raise NotImplementedError

    @abstractmethod
    def argmax(self, axis=None):
        """Indices of the maximum values along an axis.

        Parameters
        ----------
        axis : int or None, optional
            If None (default), array is flattened before computing
            index, otherwise the specified axis is used.

        Returns
        -------
        int or CArrayInterface
            Scalar with index of the maximum value for flattened array or
            CArrayInterface with indices along the given axis.

        """
        raise NotImplementedError

    @abstractmethod
    def argmin(self, axis=None):
        """Indices of the minimum values along an axis.

        Parameters
        ----------
        axis : int or None, optional
            If None (default), array is flattened before computing
            index, otherwise the specified axis is used.

        Returns
        -------
        int or CArray
            Scalar with index of the minimum value for flattened array or
            CArray with indices along the given axis.

        """
        raise NotImplementedError

    @abstractmethod
    def nanmax(self, axis=None, keepdims=True):
        """Return the maximum of an array or maximum along an axis ignoring Nans.

        When all-NaN slices are encountered a RuntimeWarning is raised
        and Nan is returned for that slice.

        Parameters
        ----------
        axis : int or None, optional
            Axis along which to operate.
            If None (default), flattened input is used.
        keepdims : bool, optional
            If this is set to True (default), the result will
            broadcast correctly against the original array.
            Otherwise resulting array is flattened.

        Returns
        -------
        scalar or CArrayInterface
            Maximum of array ignoring Nans.
            If axis is None, the result is a scalar value.
            If axis is given, the result is an array of
            dimension array.ndim - 1.

        """
        raise NotImplementedError

    @abstractmethod
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
        scalar or CArrayInterface
            Minimum of array ignoring nans.
            If axis is None, the result is a scalar value.
            If axis is given, the result is an array of
            dimension array.ndim - 1.

        """
        raise NotImplementedError

    @abstractmethod
    def nanargmax(self, axis=None):
        """Indices of the maximum values along an axis ignoring Nans.

        For all-NaN slices ValueError is raised.
        Warning: the results cannot be trusted if a slice
        contains only NaNs and infs.

        DENSE ARRAYS ONLY

        Parameters
        ----------
        axis : int or None, optional
            If None (default), array is flattened before computing
            index, otherwise the specified axis is used.

        Returns
        -------
        int or CArray
            Scalar with index of the maximum value for flattened array or
            CArrayInterface with indices along the given axis.

        """
        raise NotImplementedError

    @abstractmethod
    def nanargmin(self, axis=None):
        """Indices of the minimum values along an axis ignoring Nans

        For all-NaN slices ValueError is raised.
        Warning: the results cannot be trusted if a slice
        contains only NaNs and infs.

        Parameters
        ----------
        axis : int or None, optional
            If None (default), array is flattened before computing
            index, otherwise the specified axis is used.

        Returns
        -------
        int or CArrayInterface
            Scalar with index of the minimum value for flattened array or
            CArrayInterface with indices along the given axis.

        """
        raise NotImplementedError

    @abstractmethod
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
        float or CArrayInterface
            Returns a new array containing the mean for given axis.
            If axis=None, returns a float scalar with global average of array.

        """
        raise NotImplementedError

    @abstractmethod
    def median(self, axis=None, keepdims=True):
        """Compute the median along the specified axis.

        Given a vector V of length N, the median of V is
        the middle value of a sorted copy of V, V_sorted - i e.,
        V_sorted[(N-1)/2], when N is odd, and the average of
        the two middle values of V_sorted when N is even.

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
        float or CArray
            Returns a new array containing the median values for given
            axis or, if axis=None, return a float scalar with global
            median of array.

        """
        raise NotImplementedError

    @abstractmethod
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
        float or CArrayInterface
            Returns a new array containing the standard deviation
            values for given axis or, if axis=None, return a float
            scalar with global standard deviation of array.

        """
        raise NotImplementedError

    # ----------------- #
    # MATH ELEMENT-WISE #
    # ----------------- #

    @abstractmethod
    def sqrt(self):
        """Compute the positive square-root of an array, element-wise.

        If any array element is complex, a complex array is returned
        (and the square-roots of negative reals are calculated). If
        all of the array elements are real, so is the resulting array,
        with negative elements returning nan.

        Returns
        -------
        CArrayInterface
            A new array with the element-wise positive square-root
            of original array.

        """
        raise NotImplementedError

    @abstractmethod
    def sin(self):
        """Trigonometric sine, element-wise.

        The array elements are considered angles, in radians
        (:math:`2\\pi` rad equals 360 degrees).

        Returns
        -------
        CArrayInterface
            New array with trigonometric sine element-wise.

        """
        raise NotImplementedError

    @abstractmethod
    def cos(self):
        """Trigonometric cosine, element-wise.

        The array elements are considered angles, in radians
        (:math:`2\\pi` rad equals 360 degrees).

        Returns
        -------
        CArrayInterface
            New array with trigonometric cosine element-wise.

        """
        raise NotImplementedError

    @abstractmethod
    def exp(self):
        """Calculate the exponential of all elements in the input array.

        Returns
        -------
        CArrayInterface
            New array with element-wise exponential of current data.

        """
        raise NotImplementedError

    @abstractmethod
    def log(self):
        """Calculate the natural logarithm of all elements in the input array.

        Returns
        -------
        CArrayInterface
            New array with element-wise natural logarithm of current data.

        """
        raise NotImplementedError

    @abstractmethod
    def log10(self):
        """Calculate the base 10 logarithm of all elements in the input array.

        Returns
        -------
        CArrayInterface
            New array with element-wise base 10 logarithm of current data.

        """
        raise NotImplementedError

    @abstractmethod
    def pow(self, exp):
        """Array elements raised to powers from input exponent, element-wise.

        Raise each base in the array to the positionally-corresponding
        power in exp. exp must be broadcastable to the same shape of array.
        If exp is a scalar, works like standard ``**`` operator.

        Parameters
        ----------
        exp : CArrayInterface or scalar
            Exponent of power, can be another array or a single scalar.
            If array, must have the same shape of original array.

        Returns
        -------
        CArrayInterface
            New array with the power of current data using
            input exponents.

        """
        raise NotImplementedError

    @abstractmethod
    def normpdf(self, mu=0.0, sigma=1.0):
        """Return normal distribution function value with mean
        and standard deviation given for the current array values.

        DENSE ARRAYS ONLY

        The norm pdf is given by:

        .. math::

            y = f(x|\\mu,\\sigma) = \\frac{1}{\\sigma \\sqrt{2\\pi}}
            e^{\\frac{-(x-\\mu)^2}{2\\sigma^2}}

        The standard normal distribution has :math:`\\mu=0`
        and :math:`\\sigma=1`.

        Parameters
        ----------
        mu : float, optional
            Normal distribution mean. Default 0.0.
        sigma : float, optional
            Normal distribution standard deviation. Default 1.0.

        Returns
        -------
        CArrayInterface
            Normal distribution values.

        """
        raise NotImplementedError

    # ----- #
    # MIXED #
    # ----- #

    @abstractmethod
    def sign(self):
        """Returns element-wise sign of the array.

        The sign function returns -1 if x < 0, 0 if x == 0, 1 if x > 0.

        Returns
        -------
        CArrayInterface
            Array with sign of each element.

        """
        raise NotImplementedError

    @abstractmethod
    def diag(self, k=0):
        """Extract a diagonal from array or construct a diagonal array.

        Parameters
        ----------
        k : int, optional
            Diagonal index. Default is 0.
            Use k > 0 for diagonals above the main diagonal,
            k < 0 for diagonals below the main diagonal.

        Returns
        -------
        CArrayInterface
            The extracted diagonal or constructed diagonal dense array.
            If array is 2-Dimensional, returns its k-th diagonal.
            If array is vector_like, return a 2-D array with
             the array on the k-th diagonal.

        """
        raise NotImplementedError

    @abstractmethod
    def dot(self, array):
        """Dot product of two arrays.

        For 2-D arrays it is equivalent to matrix multiplication.
        If both arrays are dense flat (rows), it is equivalent to the
        inner product of vectors (without complex conjugation).

        Format of output array is the same of the first product argument.

        Parameters
        ----------
        array : CArrayInterface
            Second argument of dot product.

        Returns
        -------
        scalar or CArrayInterface
            Result of dot product.
            A CArrayInterface with the same format of first argument or
            scalar if out.size == 1.

        """
        raise NotImplementedError

    @abstractmethod
    def interp(self, x_data, y_data, return_left=None, return_right=None):
        """One-dimensional linear interpolation.

        DENSE FORMAT ONLY

        Returns the 1-D piecewise linear interpolant to a function
        with given values at discrete data-points.

        Parameters
        ----------
        x_data : CArrayInterface
            Flat array of floats with the x-coordinates
            of the data points, must be increasing.
        y_data : CArrayInterface
            Flat array of floats with the y-coordinates
            of the data points, same length as `x_data`.
        return_left : float, optional
            Value to return for x < x_data[0], default is y_data[0].
        return_right : float, optional
            Value to return for x > x_data[-1], default is y_data[-1].

        Returns
        -------
        CArrayInterface
            The interpolated values, same shape as x.

        """
        raise NotImplementedError

    # -------------------------------- #
    # # # # # # CLASSMETHODS # # # # # #
    # ---------------------------------#

    @classmethod
    @abstractmethod
    def empty(cls, shape, dtype=float):
        """Return a new array of given shape and type, without filling it.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new array, e.g., 2 or (2,3).
        dtype : str or dtype, optional
            The desired data-type for the array. Default is float.

        Returns
        -------
        CArrayInterface
            Array of arbitrary values with the given shape and dtype.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def zeros(cls, shape, dtype=float):
        """Return a new array of given shape and type, filled with zeros.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new array, e.g., 2 or (2,3).
        dtype : str or dtype, optional
            The desired data-type for the array. Default is float.

        Returns
        -------
        CArrayInterface
            Array of zeros with the given properties.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def ones(cls, shape, dtype=float):
        """Return a new array of given shape and type, filled with ones.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the new array, e.g., 2 or (2,3).
        dtype : str or dtype, optional
            The desired data-type for the array. Default is float.

        Returns
        -------
        CArrayInterface
            Array of ones with the given properties.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def eye(cls, n_rows, n_cols=None, k=0, dtype=float):
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

        Returns
        -------
        CArrayInterface
            An array where all elements are equal to zero, except for the
            k-th diagonal, whose values are equal to one.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def rand(cls, shape, random_state=None):
        """Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over
        the stated interval. To sample Unif[a, b), b > a multiply
        the output of rand by (b-a) and add a:

            (b - a) * rand() + a

        Parameters
        ----------
        shape : tuple of ints
            Shape of the new array.
        random_state : int or None, optional
            If int, random_state is the seed used by the
            random number generator; If None, is the seed used by np.random.

        Returns
        -------
        CArrayInterface
            Array of random floats with the given shape and format.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
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
        CArrayInterface or float
            A new array of given shape with floating-point samples
            from the standard normal distribution, or a single such
            float if no parameters were supplied.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def randint(cls, low, high=None, shape=None, random_state=None):
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

        Returns
        -------
        CArrayInterface
            Size-shaped array of random integers.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def randsample(cls, a, shape=None, replace=False, random_state=None):
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

        Returns
        -------
        CArrayInterface
            The generated random samples.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def linspace(cls, start, stop, num=50, endpoint=True):
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

        Returns
        -------
        CArrayInterface
            There are num equally spaced samples in the closed interval
            [start, stop] or the half-open interval [start, stop) (depending
            on whether endpoint is True or False).

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def arange(cls, start, stop=None, step=1, dtype=None):
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

        Returns
        -------
        CArray
            Array of evenly spaced values. For floating point arguments,
            the length of the result is ceil((stop - start)/step). Because
            of floating point overflow, this rule may result in the last
            element of out being greater than stop.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concatenate(cls, array1, array2, axis=1):
        """Concatenate a sequence of arrays along the given axis.

        The arrays must have the same shape, except in the
        dimension corresponding to axis (the second, by default).

        This function preserves input masks if available.

        Parameters
        ----------
        array1 : CArrayInterface or array_like
            First array. If array1 is not an array, a CArrayInterface will be
            created before concatenating.
        array2 : CArrayInterface or array_like
            Second array. If array2 is not an array, a CArrayInterface will be
            created before concatenating.
        axis : int or None, optional
            The axis along which the arrays will be joined. Default is 1.
            If None, both arrays are ravelled before concatenation.

        Returns
        -------
        CArrayInterface
            The concatenated array.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def comblist(cls, list_of_list, dtype=float):
        """Generate a cartesian product of list of list input.

        Parameters
        ----------
        list_of_list : list of list
            1-D arrays to form the cartesian product of.
        dtype : str or dtype
            Datatype of output array. Default float.

        Returns
        -------
        CArray
            2-D array of shape (M, len(arrays)) containing
            cartesian products between input arrays.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def meshgrid(cls, xi, indexing='xy'):
        """Return coordinate matrices from coordinate vectors.

        Make N-D coordinate arrays for vectorized evaluations of N-D
        scalar/vector fields over N-D grids, given one-dimensional
        coordinate arrays x1, x2,..., xn.

        Parameters
        ----------
        x1, x2, ..., xi : tuple of CArray or list
            1-D arrays representing the coordinates of a grid.
        indexing : {'xy', 'ij'}, optional
            Cartesian ('xy', default) or matrix ('ij') indexing of
            output. See Examples.

        Returns
        -------
        X1, X2, ..., XN : tuple of CArray
            For vectors x1, x2,..., 'xn' with lengths Ni=len(xi),
            return (N1, N2, N3,...Nn) shaped arrays if indexing='ij'
            or (N2, N1, N3,...Nn) shaped arrays if indexing='xy' with
            the elements of xi repeated to fill the matrix along the
            first dimension for x1, the second for x2 and so on.

        """
        raise NotImplementedError
