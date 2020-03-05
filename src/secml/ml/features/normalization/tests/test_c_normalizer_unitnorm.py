from secml.ml.features.tests import CPreProcessTestCases

from sklearn.preprocessing import Normalizer

from secml.array import CArray
from secml.ml.features.normalization import CNormalizerUnitNorm
from secml.optim.function import CFunction


class TestCNormalizerUnitNorm(CPreProcessTestCases):
    """Unittest for CNormalizerUnitNorm."""

    def test_norm_unitnorm(self):
        """Test for CNormalizerUnitNorm."""

        norm_type_lst = ["l1", "l2", "max"]

        def sklearn_comp(array, norm_type):
            self.logger.info("Norm type: {:}".format(norm_type))
            self.logger.info("Original array is: {:}".format(array))

            # Sklearn normalizer (requires float dtype input)
            target = CArray(Normalizer(norm=norm_type).fit_transform(
                            array.astype(float).get_data()))

            # Create our normalizer
            result = CNormalizerUnitNorm(norm=norm_type).fit_transform(array)

            self.logger.info("Correct result is:\n{:}".format(target))
            self.logger.info("Our result is:\n{:}".format(result))

            self.assert_array_almost_equal(target, result)

        for norm_type in norm_type_lst:
            sklearn_comp(self.array_dense, norm_type)
            sklearn_comp(self.array_sparse, norm_type)
            sklearn_comp(self.row_dense.atleast_2d(), norm_type)
            sklearn_comp(self.row_sparse, norm_type)
            sklearn_comp(self.column_dense, norm_type)
            sklearn_comp(self.column_sparse, norm_type)

    def test_chain(self):
        """Test a chain of preprocessors."""
        x_chain = self._test_chain(
            self.array_dense,
            ['min-max', 'pca', 'unit-norm'],
            [{'feature_range': (-5, 5)}, {}, {}]
        )

        # Expected shape is (3, 3), as pca max n_components is 4-1
        self.assertEqual((self.array_dense.shape[0],
                          self.array_dense.shape[1] - 1), x_chain.shape)

    def _test_gradient(self):
        """Check the normalizer gradient."""

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
    CPreProcessTestCases.main()
