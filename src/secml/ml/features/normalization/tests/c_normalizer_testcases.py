from secml.testing import CUnitTest
from secml.array import CArray
from secml.ml.features.tests import CPreProcessTestCases


class CNormalizerTestCases(CPreProcessTestCases):
    """Unittests interface for Normalizers."""

    def _sklearn_comp(self, array, norm_sklearn, norm, sparse=False):
        """Tests if the sklearn normalizer and our normalizer yield same result.
        """
        self.logger.info("Original array is:\n{:}".format(array))
        if sparse:
            array_sk = array.get_data()
        else:
            array_sk = array.tondarray()

        # Sklearn normalizer
        sk_norm = norm_sklearn.fit(array_sk)
        target = CArray(sk_norm.transform(array_sk))

        # Our normalizer
        n = norm.fit(array)
        result = n.transform(array)
        self.logger.info("Correct result is:\n{:}".format(target))
        self.logger.info("Our result is:\n{:}".format(result))
        self.assert_array_almost_equal(target, result)

        array_sk = array.tondarray()
        sk_norm = norm_sklearn.fit(array_sk)

        return target, result, sk_norm, n, array_sk

    def _test_chain(self, x, pre_id_list, kwargs_list, y=None):
        """Tests if preprocess chain and manual chaining yield same result."""
        x_chain = super(CNormalizerTestCases, self)._test_chain(
            x, pre_id_list, kwargs_list, y)
        self.assertEqual((self.array_dense.shape[0],
                          self.array_dense.shape[1] - 1), x_chain.shape)

        return x_chain

    def _test_chain_gradient(self, x, pre_id_list, kwargs_list, y=None):
        """Tests if gradient preprocess chain and
        gradient of manual chaining yield same result.
        """
        grad_chain = super(CNormalizerTestCases, self)._test_chain_gradient(
            x, pre_id_list, kwargs_list, y)
        self.assertEqual((self.array_dense.shape[1],), grad_chain.shape)

        return grad_chain


if __name__ == '__main__':
    CUnitTest.main()
