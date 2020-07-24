from secml.testing import CUnitTest

from secml.array import CArray
from secml.ml import CModule


class CModuleTestCases(CUnitTest):
    """Unittests interface for CPreProcess."""

    def setUp(self):
        self.array_dense = CArray([[1, 0, 0, 5],
                                   [2, 4, 0, 0],
                                   [3, 6, 0, 0]])
        self.array_sparse = CArray(self.array_dense.deepcopy(), tosparse=True)

        self.labels = CArray([0, 1, 0])

        # found bug in sklearn normalizer, see:
        # https://github.com/scikit-learn/scikit-learn/issues/16632
        # self.row_dense = CArray([-4, 0, 6])
        self.row_dense = CArray([4, 0, 6])
        self.column_dense = self.row_dense.deepcopy().T

        self.row_sparse = CArray(self.row_dense.deepcopy(), tosparse=True)
        self.column_sparse = self.row_sparse.deepcopy().T

    @staticmethod
    def _create_chain(class_type_list, kwargs_list):
        """Creates a module with other modules chained
        and a list of the same modules (not chained)."""
        chain = None  # module with preprocessing chain
        modules = []  # list of modules (not connected via preprocessing)
        for i, pre_id in enumerate(class_type_list):
            chain = CModule.create(
                pre_id, preprocess=chain, **kwargs_list[i])
            modules.append(CModule.create(pre_id, **kwargs_list[i]))
        return chain, modules

    def _test_chain(self, x, class_type_list, kwargs_list, y=None):
        """Tests if preprocess chain and manual chaining yield same result."""
        chain, modules = self._create_chain(class_type_list, kwargs_list)

        chain = chain.fit(x, y=y)
        self.logger.info("Preprocessors chain:\n{:}".format(chain))

        x_chain = chain.forward(x)
        self.logger.info("Trasformed X (chain):\n{:}".format(x_chain))

        # Train the manual chain and transform
        x_manual = x
        for module in modules:
            module.fit(x_manual, y=y)
            x_manual = module.forward(x_manual)

        self.logger.info("Trasformed X (manual):\n{:}".format(x_manual))
        self.assert_allclose(x_chain, x_manual)

        return x_chain

    def _test_chain_gradient(self, x, class_type_list, kwargs_list, y=None):
        """Tests if gradient preprocess chain and
        gradient of manual chaining yield same result."""
        chain, modules = self._create_chain(class_type_list, kwargs_list)

        chain = chain.fit(x, y=y)
        self.logger.info("module chain:\n{:}".format(chain))

        v = x[1, :]
        fwd_chain = chain.forward(v)  # this has size equal to n_outputs

        # compute gradient of the last output
        n_outputs = fwd_chain.size
        w = CArray.zeros(shape=(n_outputs,))
        w[-1] = 1
        grad_chain = chain.gradient(v, w=w)
        self.logger.info("chain.forward({:}):\n{:}".format(v, fwd_chain))
        self.logger.info("chain.gradient({:}):\n{:}".format(v, grad_chain))

        # Manually train the chain
        for module in modules:
            module.fit(x, y=y)
            x = module.forward(x)

        # test on a single point
        v_list = [v]
        for module in modules[:-1]:
            v = module.forward(v)
            v_list.append(v)

        v_list = list(reversed(v_list))
        modules = list(reversed(modules))

        grad = w
        for i, v in enumerate(v_list):
            grad = modules[i].gradient(v, w=grad)

        self.logger.info(
            "chain.gradient({:}):\n{:}".format(v, grad))
        self.assert_allclose(grad_chain, grad)

        return grad


class TestCModule(CModuleTestCases):
    def test_chain(self):
        """Test a chain of preprocessors."""
        self._test_chain(self.array_dense,
                         ['min-max', 'pca', 'min-max', 'rbf', 'svm'],
                         [{'feature_range': (-5, 5)}, {},
                          {'feature_range': (0, 1)}, {}, {}],
                         y=self.labels)

    def test_chain_gradient(self):
        """Check gradient of a chain of preprocessors."""
        self._test_chain_gradient(self.array_dense,
                                  ['min-max', 'min-max', 'rbf', 'svm'],
                                  [{'feature_range': (0, 1)},
                                   {}, {}, {}],
                                  y=self.labels)


if __name__ == '__main__':
    CUnitTest.main()
