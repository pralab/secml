"""
.. module:: _CArrayInterface
   :synopsis: Interface for array classes

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta, abstractmethod
from copy import deepcopy

from secml.core.type_utils import to_builtin


class _CArrayInterface(metaclass=ABCMeta):
    """Interface for array classes.

    For extensive definition of each method, see `secml.array.CArray`.

    """

    # ------------------------------ #
    # # # # # # PROPERTIES # # # # # #
    # -------------------------------#

    @property
    @abstractmethod
    def shape(self):
        """Shape of stored data, tuple of ints."""
        raise NotImplementedError

    @property
    @abstractmethod
    def input_shape(self):
        """Original shape of input data, tuple of ints."""
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self):
        """Size (number of elements) of array."""
        raise NotImplementedError

    @property
    @abstractmethod
    def ndim(self):
        """Number of array dimensions."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self):
        """Data-type of stored data."""
        raise NotImplementedError

    @property
    @abstractmethod
    def nnz(self):
        """Number of non-zero values in the array."""
        raise NotImplementedError

    @property
    @abstractmethod
    def nnz_indices(self):
        """Index of non-zero array elements."""
        raise NotImplementedError

    @property
    @abstractmethod
    def nnz_data(self):
        """Return non-zero array elements."""
        raise NotImplementedError

    @property
    @abstractmethod
    def T(self):
        """Transposed array data."""
        raise NotImplementedError

    # --------------------------- #
    # # # # # # CASTING # # # # # #
    # ----------------------------#

    @abstractmethod
    def tondarray(self, shape=None):
        """Return a dense numpy.ndarray representation of array."""
        raise NotImplementedError

    @abstractmethod
    def tocsr(self, shape=None):
        """Return a sparse scipy.sparse.csr_matrix representation of array."""
        raise NotImplementedError

    @abstractmethod
    def tocoo(self, shape=None):
        """Return a sparse scipy.sparse.coo_matrix representation of array."""
        raise NotImplementedError

    @abstractmethod
    def tocsc(self, shape=None):
        """Return a sparse scipy.sparse.csc_matrix representation of array."""
        raise NotImplementedError

    @abstractmethod
    def todia(self, shape=None):
        """Return a sparse scipy.sparse.dia_matrix representation of array."""
        raise NotImplementedError

    @abstractmethod
    def todok(self, shape=None):
        """Return a sparse scipy.sparse.dok_matrix representation of array."""
        raise NotImplementedError

    @abstractmethod
    def tolil(self, shape=None):
        """Return a sparse scipy.sparse.lil_matrix representation of array."""
        raise NotImplementedError

    @abstractmethod
    def tolist(self, shape=None):
        """Return the array as a (possibly nested) list."""
        raise NotImplementedError

    # ---------------------------- #
    # # # # # # INDEXING # # # # # #
    # -----------------------------#

    @abstractmethod
    def __getitem__(self, idx):
        """Return a new array with slicing/indexing result."""
        raise NotImplementedError

    def item(self):
        """Returns the single element in the array as built-in type."""
        if self.size != 1:
            raise ValueError(
                "cannot use .item(). Array has size {:}".format(self.size))
        return to_builtin(self.tondarray().ravel()[0])

    @abstractmethod
    def __setitem__(self, idx, value):
        """Set input data to slicing/indexing result."""
        raise NotImplementedError

    # ------------------------------------ #
    # # # # # # SYSTEM OVERLOADS # # # # # #
    # -------------------------------------#

    @abstractmethod
    def __add__(self, other):
        """Element-wise addition."""
        raise NotImplementedError

    @abstractmethod
    def __radd__(self, other):
        """Element-wise (inverse) addition."""
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other):
        """Element-wise subtraction."""
        raise NotImplementedError

    @abstractmethod
    def __rsub__(self, other):
        """Element-wise (inverse) subtraction."""
        raise NotImplementedError

    @abstractmethod
    def __mul__(self, other):
        """Element-wise product."""
        raise NotImplementedError

    @abstractmethod
    def __rmul__(self, other):
        """Element-wise (inverse) product."""
        raise NotImplementedError

    @abstractmethod
    def __truediv__(self, other):
        """Element-wise true division."""
        raise NotImplementedError

    @abstractmethod
    def __rtruediv__(self, other):
        """Element-wise (inverse) true division."""
        raise NotImplementedError

    @abstractmethod
    def __floordiv__(self, other):
        """Element-wise floor division (// operator)."""
        raise NotImplementedError

    @abstractmethod
    def __rfloordiv__(self, other):
        """Element-wise (inverse) floor division (// operator)."""
        raise NotImplementedError

    @abstractmethod
    def __abs__(self):
        """Returns array elements without sign."""
        raise NotImplementedError

    @abstractmethod
    def __neg__(self):
        """Returns array elements with negated sign."""
        raise NotImplementedError

    @abstractmethod
    def __pow__(self, power):
        """Element-wise power."""
        raise NotImplementedError

    @abstractmethod
    def __rpow__(self, power):
        """Element-wise (inverse) power."""
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        """Element-wise == operator."""
        raise NotImplementedError

    @abstractmethod
    def __lt__(self, other):
        """Element-wise < operator."""
        raise NotImplementedError

    @abstractmethod
    def __le__(self, other):
        """Element-wise <= operator."""
        raise NotImplementedError

    @abstractmethod
    def __gt__(self, other):
        """Element-wise > operator."""
        raise NotImplementedError

    @abstractmethod
    def __ge__(self, other):
        """Element-wise >= operator."""
        raise NotImplementedError

    @abstractmethod
    def __ne__(self, other):
        """Element-wise != operator."""
        raise NotImplementedError

    @abstractmethod
    def __bool__(self):
        """Manage 'and' and 'or' operators."""
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        """Yields array elements in raster-scan order."""
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
        """Save array data into plain text file."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, datafile, dtype=float):
        """Load array data from plain text file."""
        raise NotImplementedError

    # ----------------------------- #
    # # # # # # UTILITIES # # # # # #
    # ------------------------------#

    # ---------------- #
    # SHAPE ALTERATION #
    # ---------------- #

    @abstractmethod
    def transpose(self):
        """Returns current array with axes transposed."""
        raise NotImplementedError

    @abstractmethod
    def ravel(self):
        """Return a flattened array."""
        raise NotImplementedError

    @abstractmethod
    def flatten(self):
        """Return a flattened copy of array."""
        raise NotImplementedError

    @abstractmethod
    def atleast_2d(self):
        """Return the array with at least two dimensions."""
        raise NotImplementedError

    @abstractmethod
    def reshape(self, newshape):
        """Gives a new shape to an array without changing its data."""
        raise NotImplementedError

    @abstractmethod
    def resize(self, newshape, constant=0):
        """Return a new array with the specified shape."""
        raise NotImplementedError

    # --------------- #
    # DATA ALTERATION #
    # --------------- #

    @abstractmethod
    def astype(self, dtype):
        """Copy of the array, casted to a specified type."""
        raise NotImplementedError

    @abstractmethod
    def nan_to_num(self):
        """Replace nan with zero and inf with finite numbers."""
        raise NotImplementedError

    @abstractmethod
    def round(self, decimals=0):
        """Evenly round to the given number of decimals."""
        raise NotImplementedError

    def rint(self):
        """Round elements of the array to the nearest integer."""
        return self.round(decimals=0)

    @abstractmethod
    def ceil(self):
        """Return the ceiling of the input, element-wise."""
        raise NotImplementedError

    @abstractmethod
    def floor(self):
        """Return the floor of the input, element-wise."""
        raise NotImplementedError

    @abstractmethod
    def clip(self, c_min, c_max):
        """Clip (limit) the values in an array."""
        raise NotImplementedError

    def abs(self):
        """Returns array elements without sign."""
        return abs(self)

    @abstractmethod
    def sort(self, axis=-1, kind='quicksort', inplace=False):
        """Sort an array."""
        raise NotImplementedError

    @abstractmethod
    def argsort(self, axis=-1, kind='quicksort'):
        """Returns the indices that would sort an array."""
        raise NotImplementedError

    @abstractmethod
    def shuffle(self):
        """Modify array in-place by shuffling its contents."""
        raise NotImplementedError

    # ------------ #
    # APPEND/MERGE #
    # ------------ #

    @abstractmethod
    def append(self, array, axis=None):
        """Append values to the end of an array."""
        raise NotImplementedError

    @abstractmethod
    def repmat(self, m, n):
        """Repeat an array M x N times."""
        raise NotImplementedError

    @abstractmethod
    def repeat(self, repeats, axis=None):
        """Repeat elements of an array."""
        raise NotImplementedError

    # ---------- #
    # COMPARISON #
    # ---------- #

    @abstractmethod
    def logical_and(self, array):
        """Element-wise logical AND of array elements."""
        raise NotImplementedError

    @abstractmethod
    def logical_or(self, array):
        """Element-wise logical OR of array elements."""
        raise NotImplementedError

    @abstractmethod
    def logical_not(self):
        """Element-wise logical NOT of array elements."""
        raise NotImplementedError

    @abstractmethod
    def maximum(self, array):
        """Element-wise maximum of array elements."""
        raise NotImplementedError

    @abstractmethod
    def minimum(self, array):
        """Element-wise minimum of array elements."""
        raise NotImplementedError

    # ------ #
    # SEARCH #
    # ------ #

    @abstractmethod
    def find(self, condition):
        """Returns array elements indices depending on condition."""
        raise NotImplementedError

    @abstractmethod
    def binary_search(self, value):
        """Returns the index of each input value inside the array."""
        raise NotImplementedError

    # ------------- #
    # DATA ANALYSIS #
    # ------------- #

    @abstractmethod
    def get_nnz(self, axis=None):
        """Counts the number of non-zero values in the array."""
        raise NotImplementedError

    @abstractmethod
    def unique(self, return_index=False,
               return_inverse=False, return_counts=False):
        """Find the unique elements of an array."""
        raise NotImplementedError

    @abstractmethod
    def bincount(self, bincount=0):
        """Count the occurrences of each value of non-negative ints."""
        raise NotImplementedError

    @abstractmethod
    def norm(self, order=None):
        """Array norm."""
        raise NotImplementedError

    @abstractmethod
    def norm_2d(self, order=None, axis=None, keepdims=True):
        """Array norm."""
        raise NotImplementedError

    @abstractmethod
    def sum(self, axis=None, keepdims=True):
        """Sum of array elements over a given axis."""
        raise NotImplementedError

    @abstractmethod
    def cumsum(self, axis=None, dtype=None):
        """Return the cumulative sum of the array elements."""
        raise NotImplementedError

    @abstractmethod
    def prod(self, axis=None, dtype=None, keepdims=True):
        """Return the product of array elements over a given axis."""
        raise NotImplementedError

    @abstractmethod
    def all(self, axis=None, keepdims=True):
        """Test whether all array elements evaluate to True."""
        raise NotImplementedError

    @abstractmethod
    def any(self, axis=None, keepdims=True):
        """Test whether any array elements evaluate to True."""
        raise NotImplementedError

    @abstractmethod
    def max(self, axis=None, keepdims=True):
        """Return the maximum of an array or maximum along an axis."""
        raise NotImplementedError

    @abstractmethod
    def min(self, axis=None, keepdims=True):
        """Return the minimum of an array or minimum along an axis."""
        raise NotImplementedError

    @abstractmethod
    def argmax(self, axis=None):
        """Indices of the maximum values along an axis."""
        raise NotImplementedError

    @abstractmethod
    def argmin(self, axis=None):
        """Indices of the minimum values along an axis."""
        raise NotImplementedError

    @abstractmethod
    def nanmax(self, axis=None, keepdims=True):
        """Return the maximum ignoring Nans."""
        raise NotImplementedError

    @abstractmethod
    def nanmin(self, axis=None, keepdims=True):
        """Return the minimum ignoring Nans."""
        raise NotImplementedError

    @abstractmethod
    def nanargmax(self, axis=None):
        """Indices of the maximum values along an axis ignoring Nans."""
        raise NotImplementedError

    @abstractmethod
    def nanargmin(self, axis=None):
        """Indices of the minimum values along an axis ignoring Nans."""
        raise NotImplementedError

    @abstractmethod
    def mean(self, axis=None, dtype=None, keepdims=True):
        """Compute the arithmetic mean along the specified axis."""
        raise NotImplementedError

    @abstractmethod
    def median(self, axis=None, keepdims=True):
        """Compute the median along the specified axis."""
        raise NotImplementedError

    @abstractmethod
    def std(self, axis=None, ddof=0, keepdims=True):
        """Compute the standard deviation along the specified axis."""
        raise NotImplementedError

    @abstractmethod
    def sha1(self):
        """Calculate the sha1 hexadecimal hash of array."""
        raise NotImplementedError

    @abstractmethod
    def is_inf(self):
        """Test element-wise for positive or negative infinity."""
        raise NotImplementedError

    @abstractmethod
    def is_posinf(self):
        """Test element-wise for positive infinity."""
        raise NotImplementedError

    @abstractmethod
    def is_neginf(self):
        """Test element-wise for negative infinity."""
        raise NotImplementedError

    @abstractmethod
    def is_nan(self):
        """Test element-wise for Not a Number (NaN)."""
        raise NotImplementedError

    # ----------------- #
    # MATH ELEMENT-WISE #
    # ----------------- #

    @abstractmethod
    def sqrt(self):
        """Compute the positive square-root of an array, element-wise."""
        raise NotImplementedError

    @abstractmethod
    def sin(self):
        """Trigonometric sine, element-wise."""
        raise NotImplementedError

    @abstractmethod
    def cos(self):
        """Trigonometric cosine, element-wise."""
        raise NotImplementedError

    @abstractmethod
    def exp(self):
        """Calculate the exponential of all elements in the input array."""
        raise NotImplementedError

    @abstractmethod
    def log(self):
        """Calculate the natural logarithm of all elements."""
        raise NotImplementedError

    @abstractmethod
    def log10(self):
        """Calculate the base 10 logarithm of all elements."""
        raise NotImplementedError

    @abstractmethod
    def pow(self, exp):
        """Array elements raised to powers from input exponent."""
        raise NotImplementedError

    @abstractmethod
    def normpdf(self, mu=0.0, sigma=1.0):
        """Return normal distribution function value with mean
        and standard deviation given for the current array values."""
        raise NotImplementedError

    # ----- #
    # MIXED #
    # ----- #

    @abstractmethod
    def sign(self):
        """Returns element-wise sign of the array."""
        raise NotImplementedError

    @abstractmethod
    def diag(self, k=0):
        """Extract a diagonal from array or construct a diagonal array."""
        raise NotImplementedError

    @abstractmethod
    def dot(self, array):
        """Dot product of two arrays."""
        raise NotImplementedError

    @abstractmethod
    def interp(self, x_data, y_data, return_left=None, return_right=None):
        """One-dimensional linear interpolation."""
        raise NotImplementedError

    # -------------------------------- #
    # # # # # # CLASSMETHODS # # # # # #
    # ---------------------------------#

    @classmethod
    @abstractmethod
    def empty(cls, shape, dtype=float):
        """Return a new array of given shape and type, without filling it."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def zeros(cls, shape, dtype=float):
        """Return a new array of given shape and type, filled with zeros."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def ones(cls, shape, dtype=float):
        """Return a new array of given shape and type, filled with ones."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def eye(cls, n_rows, n_cols=None, k=0, dtype=float):
        """Return a 2-D array with ones on the diagonal and zeros elsewhere."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def rand(cls, shape, random_state=None):
        """Return random floats in the half-open interval [0.0, 1.0)."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def randn(cls, shape, random_state=None):
        """Return samples from the "standard normal" distribution."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def randuniform(cls, low=0.0, high=1.0, shape=None, random_state=None):
        """Return random samples from low (inclusive) to high (exclusive)."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def randint(cls, low, high=None, shape=None, random_state=None):
        """Return random integers from low (inclusive) to high (exclusive)."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def randsample(cls, a, shape=None, replace=False, random_state=None):
        """Generates a random sample from a given array."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def linspace(cls, start, stop, num=50, endpoint=True):
        """Return evenly spaced numbers over a specified interval."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def arange(cls, start=None, stop=None, step=1, dtype=None):
        """Return evenly spaced values within a given interval."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concatenate(cls, array1, array2, axis=1):
        """Concatenate a sequence of arrays along the given axis."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def comblist(cls, list_of_list, dtype=float):
        """Generate a cartesian product of list of list input."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def meshgrid(cls, xi, indexing='xy'):
        """Return coordinate matrices from coordinate vectors."""
        raise NotImplementedError
