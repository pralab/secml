from secml.testing import CUnitTest

from secml.array import CArray
from secml.ml.features import CPreProcess


class CPreProcessTestCases(CUnitTest):
    """Unittests interface for CPreProcess."""

    def setUp(self):

        self.array_dense = CArray([[1, 0, 0, 5],
                                   [2, 4, 0, 0],
                                   [3, 6, 0, 0]])
        self.array_sparse = CArray(self.array_dense.deepcopy(), tosparse=True)

        # found bug in sklearn normalizer, see:
        # https://github.com/scikit-learn/scikit-learn/issues/16632
        # self.row_dense = CArray([-4, 0, 6])
        self.row_dense = CArray([4, 0, 6])
        self.column_dense = self.row_dense.deepcopy().T

        self.row_sparse = CArray(self.row_dense.deepcopy(), tosparse=True)
        self.column_sparse = self.row_sparse.deepcopy().T

    @staticmethod
    def _create_chain(pre_id_list, kwargs_list):
        """Creates a preprocessor with other preprocessors chained
        and a list of the same preprocessors (not chained)"""
        chain = None
        pre_list = []
        for i, pre_id in enumerate(pre_id_list):
            chain = CPreProcess.create(
                pre_id, preprocess=chain, **kwargs_list[i])
            pre_list.append(CPreProcess.create(pre_id, **kwargs_list[i]))

        return chain, pre_list

    def _test_chain(self, x, pre_id_list, kwargs_list, y=None):
        """Tests if preprocess chain and manual chaining yield same result."""
        chain, pre_list = self._create_chain(pre_id_list, kwargs_list)

        chain = chain.fit(x, y=y)
        self.logger.info("Preprocessors chain:\n{:}".format(chain))

        x_chain = chain.transform(x)
        self.logger.info("Trasformed X (chain):\n{:}".format(x_chain))

        # Train the manual chain and transform
        x_manual = x
        for pre in pre_list:
            x_manual = pre.fit_transform(x_manual, y=y)

        self.logger.info("Trasformed X (manual):\n{:}".format(x_manual))
        self.assert_allclose(x_chain, x_manual)

        # Reverting array (if available)
        try:
            x_chain_revert = chain.inverse_transform(x_chain)
            self.logger.info("Reverted X (chain):\n{:}".format(x_chain_revert))
            self.logger.info("Original X:\n{:}".format(x))
            self.assert_array_almost_equal(x_chain_revert, x)
        except NotImplementedError:
            self.logger.info("inverse_transform not available")

        return x_chain

    def _test_chain_gradient(self, x, pre_id_list, kwargs_list, y=None):
        """Tests if gradient preprocess chain and
        gradient of manual chaining yield same result."""
        chain, pre_list = self._create_chain(pre_id_list, kwargs_list)

        chain = chain.fit(x, y=y)
        self.logger.info("Preprocessors chain:\n{:}".format(chain))

        v = x[1, :]
        grad_chain = chain.gradient(v)
        self.logger.info(
            "gradient({:}) (chain):\n{:}".format(v, grad_chain))

        # Manually compose the chain and transform
        for pre in pre_list:
            x = pre.fit_transform(x, y=y)

        v_list = [v]
        for pre in pre_list[:-1]:
            v = pre.transform(v)
            v_list.append(v)

        v_list = list(reversed(v_list))
        pre_list = list(reversed(pre_list))

        grad = None
        for i, v in enumerate(v_list):
            grad = pre_list[i].gradient(v, w=grad)

        self.logger.info(
            "gradient({:}) (manual):\n{:}".format(v, grad))
        self.assert_allclose(grad_chain, grad)

        return grad_chain


if __name__ == '__main__':
    CUnitTest.main()
