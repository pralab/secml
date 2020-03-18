from secml.testing import CUnitTest
from secml.array import CArray
from secml.ml.features.tests import CPreProcessTestCases


class CNormalizerTestCases(CPreProcessTestCases):
    """Unittests interface for CPreProcess.
    """

    def _sklearn_comp(self, array, norm_sklearn, norm, sparse=False):
        """Check if the result given by the sklearn normalizer is almost equal to the one given by our normalizer
        """
        self.logger.info("Original array is:\n{:}".format(array))
        if sparse:
            target = CArray(norm_sklearn.fit_transform(array.get_data()))
        else:
            target = CArray(norm_sklearn.fit_transform(array.tondarray()))
        # Our normalizer
        n = norm.fit(array)
        result = n.transform(array)
        self.logger.info("Correct result is:\n{:}".format(target))
        self.logger.info("Our result is:\n{:}".format(result))
        self.assert_array_almost_equal(target, result)

    def _test_chain(self, x, pre_id_list, kwargs_list, y=None):
        """Tests if preprocess chain and manual chaining yield same result.
        """
        x_chain = super(CNormalizerTestCases, self)._test_chain(x, pre_id_list, kwargs_list, y=None)
        self.assertEqual((self.array_dense.shape[0],
                          self.array_dense.shape[1] - 1), x_chain.shape)

    def _test_chain_gradient(self, x, pre_id_list, kwargs_list, y=None):
        """Tests if gradient preprocess chain and
        gradient of manual chaining yield same result.
        """
        grad_chain = super(CNormalizerTestCases, self)._test_chain_gradient(x, pre_id_list, kwargs_list, y=None)
        self.assertEqual((self.array_dense.shape[1],), grad_chain.shape)


if __name__ == '__main__':
    CUnitTest.main()
