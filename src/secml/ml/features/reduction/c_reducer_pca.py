"""
.. module:: CPCA
   :synopsis: Principal Component Analysis (PCA)

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml.array import CArray
from secml.ml.features.reduction import CReducer

__all__ = ['CPCA']


class CPCA(CReducer):
    """Principal Component Analysis (PCA).

    Properties
    ----------
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'pca'

    """
    __class_type = 'pca'

    def __init__(self, n_components=None, preprocess=None):
        """Principal Component Analysis (PCA)

        Apply a linear transformation at data, project it into
        a new spaces where total variance of data is maximized
        (axis are the eigenvector of your matrix). If number of
        components is less than number of patterns is useful for
        project data into a low dimensional space.

        This implementation uses the scipy.linalg implementation
        of the singular value reduction. It only works for
        dense arrays and is not scalable to large dimensional data.

        Parameters
        ----------
        n_components : None or int, optional
            Number of components to keep. If n_components is not set::
                n_components == min(n_samples, n_features)
            If ``0 < n_components < 1``, select the number of components
            such that the amount of variance that needs to be explained
            is greater than the percentage specified by n_components.

        Notes
        -----
        While this implementation transparently works for sparse format
        arrays too, data is converted to dense form internally and thus
        memory consumption can be high.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.features.reduction import CPCA

        >>> array = CArray([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]])
        >>> CPCA().fit_transform(array)
        CArray([[ -4.07872199e+00   2.47826647e+00   0.00000000e+00]
         [ -2.72723183e+00  -2.82960262e+00   5.55111512e-17]
         [  6.80595382e+00   3.51336152e-01  -2.22044605e-16]])

        """
        self.n_components = n_components
        self._eigenvec = None
        self._eigenval = None
        self._components = None
        self._mean = None
        self._explained_variance = None
        self._explained_variance_ratio = None

        super(CPCA, self).__init__(preprocess=preprocess)

    @property
    def eigenval(self):
        """Eigenvalues estimated from the training data."""
        return self._eigenval

    @property
    def eigenvec(self):
        """Eigenvectors estimated from the training data."""
        return self._eigenvec

    @property
    def components(self):
        """Eigenvectors of inverse training array."""
        return self._components

    @property
    def mean(self):
        """Per-feature empirical mean, estimated from the training data."""
        return self._mean

    @property
    def explained_variance(self):
        """Variance explained by each of the selected components."""
        return self._explained_variance

    @property
    def explained_variance_ratio(self):
        """Percentage of variance explained by each of the selected components.

        If n_components is None, then all components are stored and
        the sum of explained variances is equal to 1.0

        """
        return self._explained_variance_ratio

    def _fit(self, x, y=None):
        """Fit the PCA using input data.

        Parameters
        ----------
        x : array_like
            Training data, 2-Dim array like object with shape
            (n_patterns, n_features), where each row is a pattern
            of n_features columns.
        y : CArray or None, optional
            Flat array with the label of each pattern.
            Can be None if not required by the preprocessing algorithm.

        Returns
        -------
        CPCA
            Instance of the trained transformer.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.features.reduction import CPCA

        >>> array = CArray([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]])
        >>> pca = CPCA().fit(array)
        >>> pca.eigenval
        CArray([  8.39015935e+00   3.77781588e+00   1.90957046e-17])
        >>> pca.eigenvec
        CArray([[-0.48613165  0.6560051   0.57735027]
         [-0.32505126 -0.74900491  0.57735027]
         [ 0.8111829   0.09299981  0.57735027]])
        >>> pca.explained_variance
        CArray([  2.34649246e+01   4.75729760e+00   1.21548644e-34])
        >>> pca.explained_variance_ratio
        CArray([  8.31434337e-01   1.68565663e-01   4.30684173e-36])

        """
        data_carray = CArray(x).todense().atleast_2d()
        # Max number of components is the number of patterns available (rows)
        n_samples = data_carray.shape[0]
        n_features = data_carray.shape[1]

        if self.n_components is None:
            self.n_components = min(n_samples, n_features)
        else:
            if self.n_components > n_samples:
                raise ValueError("maximum number of components is {:}".format(n_samples))

        # Centering training data
        self._mean = data_carray.mean(axis=0, keepdims=False)
        data_carray -= self.mean

        # Performing training of PCA (used by KernelPCA too)
        return self._svd_train(data_carray)

    def _svd_train(self, data_carray):
        """Linear PCA training routine, used also by KernelPCA."""

        # Computing SVD reduction
        from numpy import linalg
        from sklearn.utils.extmath import svd_flip
        u, s, v = linalg.svd(data_carray.atleast_2d().tondarray(),
                             full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        u, v = svd_flip(u, v)

        eigenvec = CArray(u)
        eigenval = CArray(s)
        components = CArray(v)

        # Now we sort the eigenvalues/eigenvectors
        idx = (-eigenval).argsort(axis=None)
        eigenval = CArray(eigenval[idx])
        eigenvec = CArray(eigenvec[:, idx]).atleast_2d()
        components = CArray(components[idx, :]).atleast_2d()
        # percentage of variance explained by each component
        explained_variance = (eigenval ** 2) / (data_carray.shape[0] - 1)
        explained_variance_ratio = explained_variance / explained_variance.sum()

        if 0 < self.n_components < 1.0:
            # number of components for which the cumulated explained variance
            # percentage is superior to the desired threshold
            ratio_cumsum = explained_variance_ratio.cumsum()
            self.n_components = CArray(ratio_cumsum < self.n_components).sum() + 1

        # Consider only n_components
        self._eigenval = CArray(eigenval[:self.n_components])
        self._eigenvec = CArray(eigenvec[:, :self.n_components])
        self._components = CArray(components[:self.n_components, :])

        # storing explained variance of n_components only
        self._explained_variance = explained_variance[:self.n_components]
        self._explained_variance_ratio = explained_variance_ratio[:self.n_components]

        return self

    def _transform(self, x):
        """Apply the reduction algorithm on data.

        Parameters
        ----------
        x : array_like
            Array to be transformed. 2-D array object of shape
            (n_patterns, n_features). n_features must be equal to
            n_components parameter set before or during training.

        Returns
        --------
        CArray
            Input data mapped to PCA space.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.features.reduction import CPCA

        >>> array = CArray([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]])
        >>> pca = CPCA().fit(array)
        >>> pca.transform(CArray.concatenate(array, [4., 2., -6.], axis=0))
        CArray([[ -4.07872199e+00   2.47826647e+00   0.00000000e+00]
         [ -2.72723183e+00  -2.82960262e+00   5.55111512e-17]
         [  6.80595382e+00   3.51336152e-01  -2.22044605e-16]
         [  3.20915225e+00  -1.12968039e+00   3.29690893e+00]])

        >>> pca.transform([4., 2.])
        Traceback (most recent call last):
            ...
        ValueError: array to transform must have 3 features (columns).

        """
        if self._mean is None:
            raise ValueError("fit PCA first.")

        data_carray = CArray(x).todense().atleast_2d()
        if data_carray.shape[1] != self.mean.size:
            raise ValueError("array to transform must have {:} "
                             "features (columns).".format(self.mean.size))

        out = CArray((data_carray - self.mean).dot(self._components.T))
        return out.atleast_2d() if x.ndim >= 2 else out

    def _revert(self, x):
        """Map data back to its original space.

        Parameters
        ----------
        x : CArray
            Array to transform back to its original space.

        Returns
        --------
        CArray
            Input array mapped back to its original space.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.ml.features.reduction import CPCA

        >>> array = CArray([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]])
        >>> pca = CPCA().fit(array)
        >>> array_pca = pca.transform(array)
        >>> pca.revert(array_pca).round(6)
        CArray([[ 1. -0.  2.]
         [ 2.  5. -0.]
         [-0.  1. -9.]])

        """
        if self._mean is None:
            raise ValueError("fit PCA first.")

        data_carray = CArray(x).atleast_2d()
        if data_carray.shape[1] != self.n_components:
            raise ValueError("array to revert must have {:} "
                             "features (columns).".format(self.n_components))

        out = CArray(data_carray.dot(self._components) + self.mean)

        return out.atleast_2d() if x.ndim >= 2 else out
