import unittest
from sklearn.feature_extraction.text import TfidfTransformer

from secml.array import CArray
from secml.ml.features.normalization import CNormalizerTFIDF
from secml.ml.features.tests import CPreProcessTestCases
from secml.optim.function import CFunction


class TestCNormalizerTFIDF(CPreProcessTestCases):
    """Unittest for TestCNormalizerTFIDF."""

    norm_type_lst = [None, 'l2', 'l1', 'max']

    def test_norm_tfidf(self):
        """Test for TestCNormalizerTFIDF."""

        def sklearn_comp(array, norm):
            self.logger.info("Original array is:\n{:}".format(array))

            # Sklearn normalizer (requires float dtype input)
            array_sk = array.astype(float).tondarray()
            sk_norm = TfidfTransformer(norm=norm).fit(array_sk)

            target = CArray(sk_norm.transform(array_sk))

            # Our normalizer
            our_norm = CNormalizerTFIDF(norm).fit(array)
            result = our_norm.transform(array)

            self.logger.info("Correct result is:\n{:}".format(target))
            self.logger.info("Our result is:\n{:}".format(result))

            self.assert_array_almost_equal(target, result)

            # Testing out of range normalization
            self.logger.info("Testing out of range normalization")

            # Sklearn normalizer (requires float dtype input)
            target = CArray(sk_norm.transform(array_sk * 2))

            # Our normalizer
            result = our_norm.transform(array * 2)

            self.logger.info("Correct result is:\n{:}".format(target))
            self.logger.info("Our result is:\n{:}".format(result))

            self.assert_array_almost_equal(target, result)

        for norm_type in self.norm_type_lst:
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
            ['tf-idf', 'pca', 'tf-idf'],
            [{'norm': 'l2'}, {}, {}]
        )

        # Expected shape is (3, 3), as pca max n_components is 4-1
        self.assertEqual(
            (self.array_dense.shape[0], self.array_dense.shape[1] - 1),
            x_chain.shape)

    @unittest.skip
    def test_inverse_transform(self):
        """Check the inverse transform."""

        def transf_and_inverse(array, norm):
            self.logger.info("Original array is:\n{:}".format(array))
            self.logger.info("Considered norm :\n{:}".format(norm))

            # create our normalizer
            norm = CNormalizerTFIDF(norm=norm).fit(array)
            trans = norm.transform(array)
            orig = norm.inverse_transform(trans)

            self.assert_array_almost_equal(array, orig)

        for norm_type in self.norm_type_lst:
            transf_and_inverse(self.array_dense, norm_type)
            transf_and_inverse(self.array_sparse, norm_type)
            transf_and_inverse(self.row_dense.atleast_2d(), norm_type)
            transf_and_inverse(self.row_sparse, norm_type)
            transf_and_inverse(self.column_dense, norm_type)
            transf_and_inverse(self.column_sparse, norm_type)

    def test_gradient(self):
        """Check the normalizer gradient."""

        def compare_analytical_and_numerical_grad(array, norm_type):

            def _get_transform_component(x, y):
                trans = norm.transform(x).todense()
                return trans[y]

            norm = CNormalizerTFIDF(norm=norm_type).fit(array)

            if (norm_type == 'l1') or (norm_type == 'max'):
                # if the norm is one we are computing a sub-gradient
                decimal = 1
            else:
                decimal = 4

            # check if they are almost equal
            self.logger.info("norm: {:}".format(norm))

            # check the gradient comparing it with the numerical one
            n_feats = array.size

            for f in range(n_feats):
                self.logger.info(
                    "Compare the gradient of feature: {:}".format(f))

                # compute analytical gradient
                w = CArray.zeros(array.size)
                w[f] = 1

                an_grad = norm.gradient(array, w=w)
                self.logger.info("Analytical gradient is:\n{:}".format(an_grad))

                num_grad = CFunction(_get_transform_component).approx_fprime(
                    array.todense(), epsilon=1e-5, y=f)
                self.logger.info("Numerical gradient is:\n{:}".format(num_grad))

                self.assert_array_almost_equal(an_grad, num_grad,
                                               decimal=decimal)

        for norm_type in self.norm_type_lst:
            compare_analytical_and_numerical_grad(self.row_dense.ravel(),
                                                  norm_type=norm_type)
            compare_analytical_and_numerical_grad(self.row_sparse,
                                                  norm_type=norm_type)


if __name__ == '__main__':
    CPreProcessTestCases.main()
