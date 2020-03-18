from secml.ml.features.normalization.tests import CNormalizerTestCases
from sklearn.preprocessing import Normalizer
from secml.array import CArray
from secml.ml.features.normalization import CNormalizerUnitNorm
from secml.optim.function import CFunction


class TestCNormalizerUnitNorm(CNormalizerTestCases):
    """Unittest for CNormalizerUnitNorm.
    """
    def _sklearn_comp(self, array, norm_sklearn, norm, norm_type=None):
        self.logger.info("Norm type: {:}".format(norm_type))
        norm_sklearn = norm_sklearn(norm=norm_type)
        norm = norm(norm=norm_type)
        super(TestCNormalizerUnitNorm, self)._sklearn_comp(array, norm_sklearn,
                                                           norm, sparse=True)

    def test_norm_unitnorm(self):
        """Test for CNormalizerUnitNorm.
        """
        norm_type_lst = ["l1", "l2", "max"]

        for norm_type in norm_type_lst:
            self._sklearn_comp(self.array_dense, Normalizer,
                               CNormalizerUnitNorm, norm_type)
            self._sklearn_comp(self.array_sparse, Normalizer,
                               CNormalizerUnitNorm, norm_type)
            self._sklearn_comp(self.row_dense.atleast_2d(), Normalizer,
                               CNormalizerUnitNorm, norm_type)
            self._sklearn_comp(self.row_sparse, Normalizer,
                               CNormalizerUnitNorm, norm_type)
            self._sklearn_comp(self.column_dense, Normalizer,
                               CNormalizerUnitNorm, norm_type)
            self._sklearn_comp(self.column_sparse, Normalizer,
                               CNormalizerUnitNorm, norm_type)

    def test_chain(self):
        """Test a chain of preprocessors.
        """
        self._test_chain(self.array_dense,
                         ['min-max', 'pca', 'unit-norm'],
                         [{'feature_range': (-5, 5)}, {}, {}])
        # Expected shape is (3, 3), as pca max n_components is 4-1

    def _test_gradient(self):
        """Check the normalizer gradient.
        """

        norm_type_lst = ["l1", "l2", "max"]

        def compare_analytical_and_numerical_grad(array, norm_type):

            def _get_transform_component(x, y):
                trans = norm.transform(x).todense()
                return trans[y]

            norm = CNormalizerUnitNorm(norm=norm_type).fit(array)

            if norm_type == "l1":
                # if the norm is one we are computing a sub-gradient
                decimal = 1
            else:
                decimal = 4

            # check if they are almost equal
            self.logger.info("Norm: {:}".format(norm))

            # check the gradient comparing it with the numerical one
            n_feats = array.size

            for f in range(n_feats):
                self.logger.info(
                    "Compare the gradient of feature: {:}".format(f))

                # compute analytical gradient
                w = CArray.zeros(array.size)
                w[f] = 1

                an_grad = norm.gradient(array, w=w)
                self.logger.info("Analytical gradient is: {:}".format(
                    an_grad.todense()))

                num_grad = CFunction(_get_transform_component).approx_fprime(
                    array.todense(), epsilon=1e-5, y=f)
                self.logger.info("Numerical gradient is: {:}".format(
                    num_grad.todense()))

                self.assert_array_almost_equal(an_grad, num_grad,
                                               decimal=decimal)

        for norm_type in norm_type_lst:
            compare_analytical_and_numerical_grad(
                self.row_dense.ravel(), norm_type=norm_type)
            compare_analytical_and_numerical_grad(
                self.row_sparse, norm_type=norm_type)
            compare_analytical_and_numerical_grad(
                (100 * self.row_dense).ravel(), norm_type=norm_type)
            compare_analytical_and_numerical_grad(
                (100 * self.row_sparse), norm_type=norm_type)


if __name__ == '__main__':
    CNormalizerTestCases.main()
