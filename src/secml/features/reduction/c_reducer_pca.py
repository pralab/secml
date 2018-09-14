"""
.. module:: PCA
   :synopsis: Principal Component Analysis (PCA)

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml.array import CArray
from secml.features.reduction import CReducer

__all__ = ['CPca', 'CKernelPca']


class CPca(CReducer):
    """Principal Component Analysis (PCA)"""
    class_type = 'pca'

    def __init__(self, n_components=None):
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
        >>> from secml.features.reduction import CPca

        >>> array = CArray([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]])
        >>> CPca().train_transform(array)
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

    def train(self, data, y=None):
        """Train the PCA using input data.

        Parameters
        ----------
        data : array_like
            Training data, 2-Dim array like object with shape
            (n_patterns, n_features), where each row is a pattern
            of n_features columns.

        Returns
        -------
        trained_PCA : CPca
            Instance of the PCA trained on input data.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.features.reduction import CPca

        >>> array = CArray([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]])
        >>> pca = CPca().train(array)
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
        data_carray = CArray(data).todense().atleast_2d()
        # Max number of components is the number of patterns available (rows)
        n_samples = data_carray.shape[0]
        n_features = data_carray.shape[1]

        if self.n_components is None:
            self.n_components = min(n_samples, n_features)
        else:
            if self.n_components > n_samples:
                raise ValueError("maximum number of components is {:}".format(n_samples))

        # Centering training data
        self._mean = CArray(data_carray.mean(axis=0)).ravel()
        data_carray -= self.mean

        # Performing training of PCA (used by KernelPCA too)
        return self._svd_train(data_carray)

    def _svd_train(self, data_carray):
        """Linear PCA training routine, used also by KernelPCA."""

        # Computing SVD reduction
        from numpy import linalg
        from sklearn.utils.extmath import svd_flip
        u, s, v = linalg.svd(data_carray.atleast_2d().tondarray(), full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        u, v = svd_flip(u, v)

        eigenvec = CArray(u)
        eigenval = CArray(s)
        components = CArray(v)

        # Now we sort the eigenvalues/eigenvectors
        idx = (-eigenval).argsort()
        eigenval = CArray(eigenval[idx])
        eigenvec = CArray(eigenvec[:, idx]).atleast_2d()
        components = CArray(components[idx, :]).atleast_2d()
        # percentage of variance explained by each component
        explained_variance = (eigenval ** 2) / data_carray.shape[0]
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
        self._explained_variance = explained_variance[0:self.n_components]
        self._explained_variance_ratio = explained_variance_ratio[0:self.n_components]

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
            Input data mapped to PCA space.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.features.reduction import CPca

        >>> array = CArray([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]])
        >>> pca = CPca().train(array)
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
            raise ValueError("train PCA first.")

        data_carray = CArray(data).todense().atleast_2d()
        if data_carray.shape[1] != self.mean.size:
            raise ValueError("array to transform must have {:} features (columns).".format(self.mean.size))

        out = CArray((data_carray - self.mean).dot(self._components.T))
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
        >>> from secml.features.reduction import CPca

        >>> array = CArray([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]])
        >>> pca = CPca().train(array)
        >>> array_pca = pca.transform(array)
        >>> pca.revert(array_pca).round(6)
        CArray([[ 1. -0.  2.]
         [ 2.  5. -0.]
         [-0.  1. -9.]])

        """
        if self._mean is None:
            raise ValueError("train PCA first.")

        data_carray = CArray(data).atleast_2d()
        if data_carray.shape[1] != self.n_components:
            raise ValueError("array to revert must have {:} features (columns).".format(self.n_components))

        out = CArray(data_carray.dot(self._components) + self.mean)
        return out.atleast_2d() if data.ndim >= 2 else out


# FIXME: THERE ARE STILL FEW DIFFERENCES WITH SKLEARN IMPLEMENTATION RESULTS
class CKernelPca(CPca):
    """Kernel Principal component analysis (KPCA)"""
    class_type = 'kpca'

    def __init__(self, kernel=None, n_components=None, whiten=True):
        """Kernel Principal component analysis (KPCA).[1]_

        Non-linear dimensionality reduction through the use of kernels.
        Input data is first mapped in the kernel space and than a
        standard PCA is applied.

        Parameters
        ----------
        kernel : CKernel subclass or None
            Instance of kernel to use for PCA transformation. By default
            a 'linear' kernel is created. Kernel-specific parameters must
            be set internally in the instance.
        n_components : None or int, optional
            Number of components to keep. if n_components is not set::
                n_components == train_array.shape[0]
            otherwise::
                n_components == min(train_array.shape[0], n_components)
        whiten : bool, optional
            When True (by default) the components array is divided by n_samples
            times singular values to ensure uncorrelated outputs with unit
            component-wise variances. For KernelPCA this helps improve the
            predictive accuracy of the downstream estimators by making there
            data respect some hard-wired assumptions.

        See Also
        --------
        prlib.kernel : module with all the available kernel metrics.
        CPca : standard PCA transformation.

        References
        ----------
        .. [1] Scholkopf, B., Smola, A., & Muller, K. R. (1997).
               "Kernel principal component analysis." In Artificial
               Neural Networks-ICANN'97 (pp. 583-588). Springer Berlin
               Heidelberg.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.kernel import CKernelLinear
        >>> from secml.features.reduction import CKernelPca
        >>> array = CArray([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]])
        >>> CKernelPca(kernel=CKernelLinear()).train_transform(array)
        CArray([[ -4.07872199e+00   2.47826647e+00   5.96046448e-08]
         [ -2.72723183e+00  -2.82960262e+00   8.94069672e-08]
         [  6.80595382e+00   3.51336152e-01  -1.19209290e-07]])

        """
        super(self.__class__, self).__init__(n_components)

        # We now store the instance of desired kernel
        from secml.kernel import CKernel, CKernelLinear
        if kernel is None:
            self.kernel = CKernelLinear()
        elif not isinstance(kernel, CKernel):
            raise TypeError("kernel must be an CKernel subclass. See prlib.kernel for more informations.")
        else:
            self.kernel = kernel
        # We have to store training array to compute kernel later
        self._train_array = None
        self._whiten = whiten

    @property
    def whiten(self):
        """If True, the components array is divided by n_samples times singular values."""
        return self._whiten

    # TODO: MOVE TO SPECIFIC CLASS CKernelCenterer
    def _centerer_fit(self, kernel):
        """Fit KernelCenterer."""
        self._mean_row = CArray(kernel.sum(axis=0, keepdims=False)) / kernel.shape[0]
        self._mean_all = self._mean_row.sum() / kernel.shape[0]

        return self

    def _centerer_transform(self, kernel):
        """Center kernel matrix."""
        if self._mean_row is None or self._mean_all is None:
            raise ValueError("Centerer not trained...")

        pred_cols = CArray(kernel.sum(axis=1, keepdims=False)) / self._mean_row.size

        return kernel + self._mean_all - self._mean_row.T - pred_cols

    def train(self, data, y=None):
        """Train the Kernel PCA using input data.

        Parameters
        ----------
        data : array_like
            Training data, 2-Dim array like object with shape
            (n_patterns, n_features), where each row is a pattern
            of n_features columns.

        Returns
        -------
        trained_KPCA : CKernelPca
            Instance of the Kernel PCA trained on input data.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.kernel import CKernelLinear
        >>> from secml.features.reduction import CKernelPca
        >>> array = CArray([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]])
        >>> kpca = CKernelPca(kernel=CKernelLinear()).train(array)
        >>> kpca.eigenval
        CArray([  7.03947739e+01   1.42718928e+01   2.61554597e-15])
        >>> kpca.eigenvec
        CArray([[-0.48613165  0.6560051   0.57735027]
         [-0.32505126 -0.74900491  0.57735027]
         [ 0.8111829   0.09299981  0.57735027]])
        >>> kpca.explained_variance
        CArray([  1.65180806e+03   6.78956414e+01   2.28036023e-30])
        >>> kpca.explained_variance_ratio
        CArray([  9.60518989e-01   3.94810113e-02   1.32601926e-33])

        """
        self._train_array = CArray(data).atleast_2d()

        # We manage n_components differently from standard PCA
        if self.n_components is None:
            self.n_components = self._train_array.shape[0]
        else:
            self.n_components = min(self._train_array.shape[0], self.n_components)

        self._train_array = data

        # Centering kernel matrix
        kernel = CArray(self.kernel.k(self._train_array))
        self._centerer_fit(kernel)
        kernel = self._centerer_transform(kernel)

        return super(self.__class__, self)._svd_train(kernel)

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
            Input data mapped to Kernel PCA space.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.kernel import CKernelLinear
        >>> from secml.features.reduction import CKernelPca
        >>> array = CArray([[1., 0., 2.], [2., 5., 0.], [0., 1., -9.]])
        >>> kpca = CKernelPca(kernel=CKernelLinear()).train(array)
        >>> kpca.transform(CArray.concatenate(array, [4., 2., -6.], axis=0))
        CArray([[ -4.07872199e+00   2.47826647e+00   5.96046448e-08]
         [ -2.72723183e+00  -2.82960262e+00   8.94069672e-08]
         [  6.80595382e+00   3.51336152e-01  -1.19209290e-07]
         [  3.20915225e+00  -1.12968039e+00  -5.96046448e-08]])

        >>> kpca.transform([4., 2.])
        Traceback (most recent call last):
            ...
        ValueError: array to transform must have 3 features (columns).

        """
        data_carray = CArray(data).atleast_2d()
        if data_carray.shape[1] != self._train_array.shape[1]:
            raise ValueError("array to transform must have {:} features (columns).".format(self._train_array.shape[1]))

        # Centering array to transform using mean of training data
        # This should be done before computing kernel
        kernel = CArray(self.kernel.k(data_carray, self._train_array))
        kernel = self._centerer_transform(kernel)

        # Whitening is True by default in KernelPCA
        if self.whiten is True:
            out = CArray(
                CArray(kernel).dot(self._components.T / self.eigenval ** 0.5))
            out.nan_to_num()  # Remove nans (if any component/eigval is null)
        else:
            out = CArray(CArray(kernel).dot(self._components.T))

        return out.atleast_2d() if data.ndim >= 2 else out

    def revert(self, data):
        """Map data back to its original space."""
        raise NotImplementedError("revert method is not available for {:}".format(self.__class__.__name__))
