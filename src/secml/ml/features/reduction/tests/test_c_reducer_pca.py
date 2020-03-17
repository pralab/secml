from secml.ml.features.tests import CPreProcessTestCases

from secml.array import CArray
from secml.ml.features.reduction import CPCA
from sklearn.decomposition import PCA


class TestCPca(CPreProcessTestCases):
    """Unittests for CPCA."""

    def test_pca(self):
        """Test for PCA. This compares sklearn equivalent to our method."""

        # Few test cases involve an all-zero column,
        # so PCA will trigger a 0/0 warning
        self.logger.filterwarnings(
            action='ignore',
            message='invalid value encountered in true_divide',
            category=RuntimeWarning
            )
        self.logger.filterwarnings(
            action='ignore',
            message='invalid value encountered in divide',
            category=RuntimeWarning
            )

        def sklearn_comp(array):
            self.logger.info("Original array is:\n{:}".format(array))

            # Sklearn normalizer
            sklearn_pca = PCA().fit(array.tondarray())
            target = CArray(sklearn_pca.transform(array.tondarray()))
            # Our normalizer
            pca = CPCA().fit(array)
            result = pca.transform(array)

            self.logger.info("Sklearn result is:\n{:}".format(target))
            self.logger.info("Result is:\n{:}".format(result))

            self.assert_array_almost_equal(result, target)

            original = pca.inverse_transform(result)

            self.assert_array_almost_equal(original, array)

        sklearn_comp(self.array_dense)
        sklearn_comp(self.array_sparse)
        sklearn_comp(self.row_dense.atleast_2d())
        sklearn_comp(self.row_sparse)
        sklearn_comp(self.column_dense)
        sklearn_comp(self.column_sparse)

    def test_chain(self):
        """Test a chain of preprocessors."""
        x_chain = self._test_chain(
            self.array_dense,
            ['min-max', 'unit-norm', 'pca'],
            [{'feature_range': (-5, 5)}, {}, {}]
        )

        # Expected shape is (3, 3), as pca max n_components is 4-1
        self.assertEqual((self.array_dense.shape[0],
                          self.array_dense.shape[1] - 1), x_chain.shape)

    # TODO: ADD TEST FOR GRADIENT (WHEN IMPLEMENTED)


if __name__ == '__main__':
    CPreProcessTestCases.main()
