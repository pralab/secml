"""
.. module:: LDA
   :synopsis: Linear Discriminant Analysis (LDA)

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml.array import CArray
from secml.features.reduction import CReducer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class CLda(CReducer):
    """Linear Discriminant Analysis (LDA)"""
    class_type = 'lda'

    def __init__(self, n_components=None):
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
        >>> from secml.features.reduction import CLda

        >>> ds = CDataset([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]], [1,1,2])
        >>> CLda().train_transform(ds.X, ds.Y)
        CArray([[-4.07872199]
         [-2.72723183]
         [ 6.80595382]])

        """
        self.n_components = n_components
        self._eigenvec = None
        self._mean = None
        self._scalings = None
        self._classes = None
        self._lda = None

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

    def train(self, data, targets):
        """Train the LDA using input data.

        Parameters
        ----------
        data : array_like
            Training data, 2-Dim array like object with shape
            (n_patterns, n_features), where each row is a pattern
            of n_features columns.
        target : array_like
            Flat dense array of shape (data.shape[0], ) with the
            labels corresponding to each data's pattern.

        Returns
        -------
        trained_LDA : CLda
            Instance of the LDA trained on input data.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.data import CDataset
        >>> from secml.features.reduction import CLda

        >>> ds = CDataset([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]], [1,1,2])
        >>> lda = CLda().train(ds.X, ds.Y)
        >>> lda.eigenvec
        CArray([[ 0.47140452]
        [ 0.0942809 ]
         [-0.23570226]])

        """
        data_carray = CArray(data).todense().atleast_2d()
        targets = CArray(targets)

        self._classes = targets.unique()

        if self.n_components is None:
            self.n_components = (self._classes.size - 1)
        else:
            if self.n_components > (self.classes.size - 1):
                raise ValueError("Maximum number of components is {:}".format(self.classes.size - 1))

        self._lda = LinearDiscriminantAnalysis(n_components=self.n_components)
        self._lda.fit(data_carray.tondarray(), targets.tondarray())
        self._eigenvec = CArray(self._lda.scalings_)
        self._mean = CArray(self._lda.xbar_)
        return self

    def transform(self, data):
        """Apply the reduction algorithm on data.

        Parameters
        ----------
        data : array_like
            Training data, 2-Dim array like object with shape
            (n_patterns, n_features), where each row is a pattern
            of n_features columns. n_features must be equal to
            n_components parameter set before or during training.

        Returns
        --------
        data_mapped : CArray
            Input data mapped to LDA space.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.data import CDataset
        >>> from secml.features.reduction import CLda

        >>> ds = CDataset([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]], [1,1,2])
        >>> lda = CLda().train(ds.X, ds.Y)
        >>> lda.transform(CArray.concatenate(ds.X, [4., 2., -6.], axis=0))
        CArray([[-1.20993827]
        [ 0.20427529]
        [ 1.00566298]
        [ 2.27845518]])

        >>> lda.transform([4., 2.])
        Traceback (most recent call last):
            ...
        ValueError: array to transform must have 3 features (columns).

        """
        if self.mean is None:
            raise ValueError("train LDA first.")

        data_carray = CArray(data).todense().atleast_2d()
        if data_carray.shape[1] != self.mean.size:
            raise ValueError("array to transform must have {:} features (columns).".format(self.mean.size))

        out = CArray(self._lda.transform(data_carray.tondarray()))
        return out.atleast_2d() if data.ndim >= 2 else out

    def revert(self, data):
        """Map data back to its original space.

        Parameters
        ----------
        data : array_like
            Array to transform back to its original space.

        Returns
        --------
        data_origin : CArray
            Input array mapped back to its original space.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.data import CDataset
        >>> from secml.features.reduction import CLda

        >>> ds = CDataset([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]], [1,1,2])
        >>> lda = CLda().train(ds.X, ds.Y)
        >>> array_lda = lda.transform(ds.X)
        >>> lda.revert(array_lda)
        CArray([[ 0.42962963  1.88592593 -2.04814815]
        [ 1.0962963   2.01925926 -2.38148148]
        [ 1.47407407  2.09481481 -2.57037037]])
        """
        if self._mean is None:
            raise ValueError("train LDA first.")

        data_carray = CArray(data).atleast_2d()
        if data_carray.shape[1] != self.n_components:
            raise ValueError("array to revert must have {:} features (columns).".format(self.n_components))

        out = CArray(data_carray.dot(self.eigenvec.T) + self.mean)
        return out.atleast_2d() if data.ndim >= 2 else out
