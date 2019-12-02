"""
.. module:: CLDA
   :synopsis: Linear Discriminant Analysis (LDA)

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from secml.array import CArray
from secml.ml.features.reduction import CReducer
from secml.utils.mixed_utils import check_is_fitted


class CLDA(CReducer):
    """Linear Discriminant Analysis (LDA).

    Parameters
    ----------
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'lda'

    """
    __class_type = 'lda'

    def __init__(self, n_components=None, preprocess=None):
        """Linear Discriminant Analysis (LDA)

        A classifier with a linear decision boundary, generated
        by fitting class conditional densities to the data and
        using Bayes' rule.

        The model fits a Gaussian density to each class, assuming
        that all classes share the same covariance matrix.

        The fitted model can also be used to reduce the dimensionality
        of the input by projecting it to the most discriminative
        directions.

        Parameters
        ----------
        n_components : None or int, optional
            Number of components to keep. If n_components is not set than
            (number of data classes - 1) is used.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.data import CDataset
        >>> from secml.ml.features.reduction import CLDA

        >>> ds = CDataset([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]], [1,1,2])
        >>> CLDA().fit_transform(ds.X, ds.Y)
        CArray(3, 1)(dense: [[-1.209938] [ 0.204275] [ 1.005663]])

        """
        self.n_components = n_components
        self._eigenvec = None
        self._mean = None
        self._scalings = None
        self._classes = None
        self._lda = None

        super(CLDA, self).__init__(preprocess=preprocess)

    @property
    def eigenvec(self):
        """Eigenvectors estimated from the training data.
           Is a matrix of shape:  n_eigenvectors * n_features."""
        return self._eigenvec

    @property
    def mean(self):
        """Per-feature empirical mean, estimated from the training data."""
        return self._mean

    @property
    def classes(self):
        """Unique targets used for training."""
        return self._classes

    @property
    def lda(self):
        """Trained sklearn LDA transformer."""
        return self._lda

    def _check_is_fitted(self):
        """Check if the preprocessor is trained (fitted).

        Raises
        ------
        NotFittedError
            If the preprocessor is not fitted.

        """
        check_is_fitted(self, ['_lda', 'mean'])

    def _fit(self, x, y):
        """Fit the LDA using input data.

        Parameters
        ----------
        x : CArray
            Training data, 2-Dim array like object with shape
            (n_patterns, n_features), where each row is a pattern
            of n_features columns.
        y : CArray
            Flat array with the label of each pattern.

        Returns
        -------
        trained_LDA : CLDA
            Instance of the trained transformer.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.data import CDataset
        >>> from secml.ml.features.reduction import CLDA

        >>> ds = CDataset([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]], [1,1,2])
        >>> lda = CLDA().fit(ds.X, ds.Y)
        >>> lda.eigenvec
        CArray(3, 1)(dense: [[ 0.471405] [ 0.094281] [-0.235702]])

        """
        data_carray = CArray(x).todense().atleast_2d()
        targets = CArray(y)

        self._classes = targets.unique()

        if self.n_components is None:
            self.n_components = (self._classes.size - 1)
        else:
            if self.n_components > (self.classes.size - 1):
                raise ValueError("Maximum number of components is {:}"
                                 "".format(self.classes.size - 1))

        self._lda = LinearDiscriminantAnalysis(n_components=self.n_components)
        self._lda.fit(data_carray.tondarray(), targets.tondarray())
        self._eigenvec = CArray(self._lda.scalings_)
        self._mean = CArray(self._lda.xbar_)

        return self

    def _forward(self, x):
        """Apply the reduction algorithm on data.

        Parameters
        ----------
        x : CArray
            Array to be transformed. 2-D array object of shape
            (n_patterns, n_features). n_features must be equal to
            n_components parameter set before or during training.

        Returns
        --------
        CArray
            Input data mapped to LDA space.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.data import CDataset
        >>> from secml.ml.features.reduction import CLDA

        >>> ds = CDataset([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]], [1,1,2])
        >>> lda = CLDA().fit(ds.X, ds.Y)
        >>> lda.transform(CArray.concatenate(ds.X, [4., 2., -6.], axis=0))
        CArray(4, 1)(dense: [[-1.209938] [ 0.204275] [ 1.005663] [ 2.278455]])

        >>> lda.transform([4., 2.])
        Traceback (most recent call last):
            ...
        ValueError: array to transform must have 3 features (columns).

        """
        data_carray = CArray(x).todense().atleast_2d()
        if data_carray.shape[1] != self.mean.size:
            raise ValueError("array to transform must have {:} features "
                             "(columns).".format(self.mean.size))

        out = CArray(self._lda.transform(data_carray.tondarray()))
        return out.atleast_2d() if x.ndim >= 2 else out
