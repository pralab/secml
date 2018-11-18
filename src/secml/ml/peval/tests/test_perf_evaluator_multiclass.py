import unittest
from secml.utils import CUnitTest

from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.kernel import CKernel
from secml.data.splitter import CDataSplitter
from secml.data.loader import CDLRandom


class TestCPerfEvaluatorMulticlass(CUnitTest):
    """Unit test for CPerfEvaluatorXValMulticlass."""

    def setUp(self):

        # Create dummy dataset (we want a test different from train)
        loader = CDLRandom(random_state=50000)
        self.training_dataset = loader.load()
        self.test_dataset = loader.load()

        # CREATE CLASSIFIERS
        kernel = CKernel.create('rbf')
        self.svm = CClassifierSVM(kernel=kernel)
        self.svm.verbose = 1

        self.logger.info("Using kernel {:}".format(self.svm.kernel.class_type))

    def test_params_multiclass(self):
        """Parameter estimation for multiclass classifiers."""
        # Create dummy dataset (we want a test different from train)
        tr = CDLRandom(n_classes=4, n_clusters_per_class=1,
                       random_state=50000).load()

        kernel = CKernel.create('rbf')
        multiclass = CClassifierMulticlassOVA(
            CClassifierSVM, C=1, kernel=kernel)
        multiclass.verbose = 1

        xval_parameters = {'C': [1, 10, 100], 'kernel.gamma': [0.1, 1]}

        # DO XVAL FOR CHOOSE BEST PARAMETERS
        xval_splitter = CDataSplitter.create(
            'kfold', num_folds=3, random_state=50000)

        # Set the best parameters inside the classifier
        best_params = multiclass.estimate_parameters(
            tr, xval_parameters, xval_splitter, 'accuracy',
            perf_evaluator='xval_multiclass', n_jobs=1)

        self.logger.info(
            "Multiclass SVM has now the following parameters: {:}".format(
                multiclass.get_params()))

        for clf_idx, clf in enumerate(multiclass.binary_classifiers):
            self.assertEqual(clf.C, (1.0, 1.0, 10.0, 10.0)[clf_idx])
            self.assertEqual(clf.kernel.gamma, (0.1, 0.1, 0.1, 0.1)[clf_idx])

        # Final test: train using best parameters
        multiclass.train(tr)

        for clf_idx, clf in enumerate(multiclass.binary_classifiers):
            for param in best_params:
                self.assertEqual(clf.get_params()[param],
                                 best_params[param][clf_idx])


if __name__ == '__main__':
    unittest.main()
