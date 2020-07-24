from secml.testing import CUnitTest
from secml.array import CArray
from secml.ml.features.tests import CPreProcessTestCases


class CNormalizerTestCases(CPreProcessTestCases):
    """Unittests interface for Normalizers."""

    def _sklearn_comp(self, array, norm_sklearn, norm, sparse=False):
        """Compare scikit-learn normalizer with our implementation.

        Parameters
        ----------
        array : CArray
        norm_sklearn
            Scikit-learn normalizer (from `sklearn.preprocessing`).
        norm : CNormalizer
        sparse : bool, optional
            If False (default) sklearn normalizer only supports dense data.

        Returns
        -------
        norm_sklearn
            Trained Scikit-learn normalizer (from `sklearn.preprocessing`).
        norm : CNormalizer
            Trained normalizer.

        """
        self.logger.info("Original array is:\n{:}".format(array))

        array_sk = array.get_data() if sparse is True else array.tondarray()

        # Sklearn normalizer
        norm_sklearn.fit(array_sk)
        transform_sklearn = CArray(norm_sklearn.transform(array_sk))

        # Our normalizer
        norm.fit(array)
        transform = norm.transform(array)

        self.logger.info("sklearn result is:\n{:}".format(transform_sklearn))
        self.logger.info("Our result is:\n{:}".format(transform))

        self.assert_array_almost_equal(transform_sklearn, transform)

        return norm_sklearn, norm

    def _test_chain(self, x, class_type_list, kwargs_list, y=None):
        """Tests if preprocess chain and manual chaining yield same result."""
        x_chain = super(CNormalizerTestCases, self)._test_chain(
            x, class_type_list, kwargs_list, y)

        self.assertEqual((self.array_dense.shape[0],
                          self.array_dense.shape[1] - 1), x_chain.shape)

        return x_chain

    def _test_chain_gradient(self, x, class_type_list, kwargs_list, y=None):
        """Tests if gradient preprocess chain and
        gradient of manual chaining yield same result."""
        grad_chain = super(CNormalizerTestCases, self)._test_chain_gradient(
            x, class_type_list, kwargs_list, y)

        self.assertEqual((self.array_dense.shape[1],), grad_chain.shape)

        return grad_chain


if __name__ == '__main__':
    CUnitTest.main()
