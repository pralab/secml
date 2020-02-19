from secml.ml.features.tests import CPreProcessTestCases

from sklearn.feature_extraction.text import TfidfTransformer
from secml.optim.function import CFunction

from secml.array import CArray
from secml.ml.features.normalization import CNormalizerTFIDF


class TestCNormalizerTFIDF(CPreProcessTestCases):
    """Unittest for TestCNormalizerTFIDF."""

    norms = [None, 'l2']

    def test_norm_minmax(self):
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

        for norm in self.norms:
            sklearn_comp(self.array_dense, norm)
            sklearn_comp(self.array_sparse, norm)
            sklearn_comp(self.row_dense.atleast_2d(), norm)
            sklearn_comp(self.row_sparse, norm)
            sklearn_comp(self.column_dense, norm)
            sklearn_comp(self.column_sparse, norm)

    def test_chain(self):
        """Test a chain of preprocessors."""
        x_chain = self._test_chain(
            self.array_dense,
            ['tf-idf', 'pca', 'tf-idf'],
            [{'norm': 'l2'}, {}, {}]
        )

        # Expected shape is (3, 3), as pca max n_components is 4-1
        self.assertEqual((self.array_dense.shape[0],
                          self.array_dense.shape[1]-1), x_chain.shape)

    def test_inverse_transform(self):
        """Check the inverse transform."""

        def transf_and_inverse(array, norm):

            self.logger.info("Original array is:\n{:}".format(array))
            self.logger.info("Considered norm :\n{:}".format(norm))

            # Our normalizer
            our_norm = CNormalizerTFIDF(norm=norm).fit(array)
            trans = our_norm.transform(array)
            orig = our_norm.inverse_transform(trans)

            self.assert_array_almost_equal(array, orig)

        for norm in self.norms:
            transf_and_inverse(self.array_dense, norm)
            transf_and_inverse(self.array_sparse, norm)
            transf_and_inverse(self.row_dense.atleast_2d(), norm)
            transf_and_inverse(self.row_sparse, norm)
            transf_and_inverse(self.column_dense, norm)
            transf_and_inverse(self.column_sparse, norm)


    def test_gradient(self):
        """Check the normalizer gradient."""

        def compare_analytical_and_numerical_grad(array, norm):

            def _get_transform_component(x,y):
                trans = norm.transform(x).todense()
                return trans[y]

            norm = CNormalizerTFIDF(norm=norm).fit(array)

            # check the gradient comparing it with the numerical one
            n_feats = array.size

            for f in range(n_feats):

                self.logger.info("Compare the gradient of feature:\n{"
                                 ":}".format(f))

                # compute analytical gradient
                w = CArray.zeros(array.size)
                w[f] = 1
                an_grad = norm.gradient(array, w=w)
                self.logger.info("analytical gradient is:\n{:}".format(
                    an_grad))

                num_grad = CFunction(
                    _get_transform_component).approx_fprime(array.todense(),
                                                            epsilon=1e-5, y=f)
                self.logger.info("numerical gradient is:\n{:}".format(
                    num_grad))

                # check if they are almost equal
                self.assert_array_almost_equal(an_grad, num_grad, decimal=4)

        for norm in self.norms:
            compare_analytical_and_numerical_grad(self.row_dense.ravel(),
                                                  norm = norm)
            compare_analytical_and_numerical_grad(self.row_sparse, norm = norm)


if __name__ == '__main__':
    CPreProcessTestCases.main()


