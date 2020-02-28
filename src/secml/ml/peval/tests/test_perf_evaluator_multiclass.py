import unittest
from secml.testing import CUnitTest

from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.kernels import CKernel
from secml.data.splitter import CDataSplitter
from secml.data.loader import CDLRandom


class TestCPerfEvaluatorMulticlass(CUnitTest):
    """Unit test for CPerfEvaluatorXValMulticlass."""

    def setUp(self):

        # Create dummy dataset (we want a test different from train)
        self.tr = CDLRandom(n_classes=4, n_clusters_per_class=1,
                            random_state=50000).load()
        self.ts = CDLRandom(n_classes=4, n_clusters_per_class=1,
                            random_state=10000).load()

    def _run_multiclass(self, multiclass, xval_params, expected_best):

        xval_splitter = CDataSplitter.create(
            'kfold', num_folds=3, random_state=50000)

        # Set the best parameters inside the classifier
        best_params = multiclass.estimate_parameters(
            self.tr, xval_params, xval_splitter, 'accuracy',
            perf_evaluator='xval-multiclass', n_jobs=1)

        self.logger.info(
            "Multiclass SVM has now the following parameters: {:}".format(
                multiclass.get_params()))

        for clf_idx, clf in enumerate(multiclass._binary_classifiers):
            self.assertEqual(
                clf.C, expected_best['C'][clf_idx])
            self.assertEqual(
                clf.kernel.gamma, expected_best['kernel.gamma'][clf_idx])

        # Final test: fit using best parameters
        multiclass.fit(self.tr)

        for clf_idx, clf in enumerate(multiclass._binary_classifiers):
            for param in best_params:
                self.assertEqual(clf.get_params()[param],
                                 best_params[param][clf_idx])

    def test_params_multiclass(self):
        """Parameter estimation for multiclass classifiers."""
        kernel = CKernel.create('rbf')
        multiclass = CClassifierMulticlassOVA(
            CClassifierSVM, C=1, kernel=kernel)
        multiclass.verbose = 1

        xval_parameters = {'C': [1, 10, 100], 'kernel.gamma': [0.1, 1]}

        expected = {'C': [1.0, 1.0, 10.0, 10.0],
                    'kernel.gamma': [0.1, 0.1, 0.1, 0.1]}

        self._run_multiclass(multiclass, xval_parameters, expected)

        self.logger.info("Testing with preprocessor")

        kernel = CKernel.create('rbf')
        multiclass = CClassifierMulticlassOVA(
            CClassifierSVM, C=1, kernel=kernel, preprocess='min-max')
        multiclass.verbose = 1

        xval_parameters = {'C': [1, 10, 100], 'kernel.gamma': [0.1, 1]}

        expected = {'C': [100, 10, 10, 1],
                    'kernel.gamma': [0.1, 0.1, 0.1, 0.1]}

        self._run_multiclass(multiclass, xval_parameters, expected)


if __name__ == '__main__':
    unittest.main()
