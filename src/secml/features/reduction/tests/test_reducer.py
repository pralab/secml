"""
This is the class for testing CArray

@author: Marco Melis
@author: Ambra Demontis

When adding a test method please use method to test name plus 'test_' as suffix.
As first test method line use self.logger.info("UNITTEST - CLASSNAME - METHODNAME")

"""
import unittest

from secml.utils import CUnitTest
from secml.array import CArray
from secml.features.reduction import CPca, CKernelPca, CLda
from sklearn.decomposition import PCA, KernelPCA
from secml.figure import CFigure


class TestArrayDecomposition(CUnitTest):
    def setUp(self):
        """Code to run before each test."""
        self.array_dense = CArray([[1, 0, 0, 5],
                                   [2, 4, 0, 0],
                                   [3, 6, 0, 0]])
        self.array_sparse = CArray(self.array_dense.deepcopy(), tosparse=True)

        self.row_dense = CArray([4, 0, 6])
        self.column_dense = self.row_dense.deepcopy().T

        self.row_sparse = CArray(self.row_dense.deepcopy(), tosparse=True)
        self.column_sparse = self.row_sparse.deepcopy().T

    def test_pca(self):
        """Test for PCA. This compares sklearn equivalent to our method."""

        def sklearn_comp(array):
            self.logger.info("Original array is:\n{:}".format(array))

            # Sklearn normalizer
            target = CArray(PCA().fit_transform(array.tondarray()))
            # Our normalizer
            pca = CPca().train(array)
            result = pca.transform(array)

            self.logger.info("Sklearn result is:\n{:}".format(target))
            self.logger.info("Result is:\n{:}".format(result))

            self.assertFalse((result.round(5) != target.round(5)).any(),
                             "PraLib and sklearn results are different.")

            self.logger.info("Correct result is:\n{:}".format(result))

            original = pca.revert(result)

            self.assertFalse((original.round(6) != array).any(),
                             "\n{:}\nis different from original\n{:}".format(original.round(6), array))

        sklearn_comp(self.array_dense)
        sklearn_comp(self.array_sparse)
        sklearn_comp(self.row_dense.atleast_2d())  # We manage flat vectors differently from numpy/sklearn
        sklearn_comp(self.row_sparse)
        sklearn_comp(self.column_dense)
        sklearn_comp(self.column_sparse)

    def test_kernelpca(self):
        """Test for Kernel PCA. This compares sklearn equivalent to our method."""

        from secml.kernel import CKernelLinear, CKernelRBF, CKernelPoly

        def sklearn_comp(array):
            kernels = [CKernelLinear(),
                       CKernelRBF(gamma=1.0 / array.shape[1]),
                       CKernelPoly(gamma=1.0 / array.shape[1], degree=3)]

            for kernel in kernels:
                self.logger.info("Original array is:\n{:}".format(array))

                self.logger.info("Using {:}".format(kernel.__class__.__name__))

                # Sklearn normalizer
                sklearn_pca = KernelPCA(kernel=kernel.class_type,
                                        n_components=array.shape[0],
                                        remove_zero_eig=False).fit(array.tondarray())
                target = CArray(sklearn_pca.transform(array.tondarray()))
                target.nan_to_num()  # Removing nans
                # Our normalizer
                pca = CKernelPca(kernel=kernel).train(array)
                result = pca.transform(array)

                self.logger.info("Sklearn result is:\n{:}".format(target))
                self.logger.info("Result is:\n{:}".format(result))

                self.assertFalse((abs(result).round(5) != abs(target).round(5)).any(),
                                 "PraLib and sklearn results are different.")

        sklearn_comp(self.array_dense)
        sklearn_comp(self.array_sparse)
        sklearn_comp(self.row_dense.atleast_2d())  # We manage flat vectors differently from numpy/sklearn
        sklearn_comp(self.row_sparse)
        sklearn_comp(self.column_dense)
        sklearn_comp(self.column_sparse)

    def test_lda(self):
        """Test for LDA. Check LDA Result Graphically.

        Apply Lda to Sklearn Iris Dataset and compare it with
        "Linear Discriminant Analysis bit by bit" by Sebastian Raschka
        http://sebastianraschka.com/Articles/2014_python_lda.html
        into the plot we must see approximatively:
        x axes: from -2 to -1 virginica, from -1 to 0 versicolor, from 1 to 2,3 setosa
        y axes: from -1 to -1 virginica, from -1 to 0.5 versicolor, from -1 to 1 setosa

        """
        from sklearn.datasets import load_iris

        iris_db = load_iris()
        patterns = CArray(iris_db.data)
        labels = CArray(iris_db.target)

        lda = CLda()
        lda.train(patterns, labels)
        # store dataset reduced with pca
        red_dts = lda.train_transform(patterns, labels)

        fig = CFigure(width=10, markersize=8)
        fig.sp.scatter(red_dts[:, 0].ravel(),
                       red_dts[:, 1].ravel(),
                       c=labels)
        fig.show()

        # revert transformation
        reverted_dts = lda.revert(red_dts)

        self.logger.info("the norm of the difference between original and "
                         "reverted matrix is {:}".format(
                            (patterns - reverted_dts).norm_2d()))


if __name__ == '__main__':
    unittest.main()
