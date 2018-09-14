import unittest
from secml.utils import CUnitTest

import sklearn.metrics as skm
import numpy as np

from secml.classifiers import CClassifierSVM
from secml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.kernel import CKernel
from secml.data.splitter import CDataSplitter
from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.peval import CPerfEvaluatorXVal
from secml.peval.metrics import CMetric
from secml.core.constants import nan

class TestCPerfEvaluator(CUnitTest):
    """Unit test for CKernel."""

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

    def test_parameters_setting(self):

        # Changing default parameters to be sure are not used
        self.svm.set_params({'C': 25, 'kernel.gamma': 1e-1})

        xval_parameters = {'C': [1, 10, 100], 'kernel.gamma': [1, 50]}

        # DO XVAL FOR CHOOSE BEST PARAMETERS
        xval_splitter = CDataSplitter.create('kfold', num_folds=5, random_state=50000)

        # Set the best parameters inside the classifier
        self.svm.estimate_parameters(self.training_dataset, xval_parameters,
                                     xval_splitter, 'accuracy', n_jobs=2)

        self.logger.info(
            "SVM has now the following parameters: {:}".format(
                self.svm.get_params()))

        self.assertEqual(self.svm.get_params()['C'], 1)
        self.assertEqual(self.svm.get_params()['kernel.gamma'], 50)

        # Now we compare the parameters chosen before with a new evaluator
        perf_eval = CPerfEvaluatorXVal(
            xval_splitter, CMetric.create('accuracy'))
        perf_eval.verbose = 1

        best_params, best_score = perf_eval.evaluate_params(
            self.svm, self.training_dataset, xval_parameters, n_jobs=2)

        for param in xval_parameters:
            self.logger.info(
                "Best '{:}' is: {:}".format(param, best_params[param]))
            self.assertEqual(best_params[param],
                             self.svm.get_params()[param])

        self.svm.verbose = 0

        parameters_combination = [
            [1, 1], [1, 50], [10, 1], [10, 50], [100, 1], [100, 50]]
        par_comb_score = CArray.zeros(len(parameters_combination))
        for comb in xrange(len(parameters_combination)):

            this_fold_score = []
            num_xval_fold = len(xval_splitter.tr_idx)

            for f in xrange(num_xval_fold):

                self.svm.set("C", parameters_combination[comb][0])
                self.svm.kernel.gamma = parameters_combination[comb][1]

                self.svm.train(
                    self.training_dataset[xval_splitter.tr_idx[f], :])

                this_fold_predicted = self.svm.classify(
                    self.training_dataset[
                        xval_splitter.ts_idx[f], :].X)[0].ravel()

                this_fold_accuracy = skm.accuracy_score(
                    self.training_dataset[
                        xval_splitter.ts_idx[f], :].Y.get_data(),
                    this_fold_predicted.get_data())
                this_fold_score.append(this_fold_accuracy)

            par_comb_score[comb] = (np.mean(this_fold_score))
            self.logger.info(
                "this fold mean: {:}".format(par_comb_score[comb]))

        max_combination_score = par_comb_score.max()
        better_param_comb = parameters_combination[par_comb_score.argmax()]
        self.logger.info("max combination score founded here: {:}".format(
            max_combination_score))
        self.logger.info("max comb score founded during xval {:}".format(
            best_score))

        self.assertEqual(max_combination_score,
                         best_score)

        # set parameters found by xval and check if are the same chosen here
        self.logger.info("the parameters selected by own xval are:")
        self.svm.set_params(best_params)
        self.logger.info("C: {:}".format(self.svm.C))
        self.logger.info("kernel.gamma: {:}".format(self.svm.kernel.gamma))
        # check c
        self.assertEqual(better_param_comb[0], self.svm.C)
        # check gamma
        self.assertEqual(better_param_comb[1], self.svm.kernel.gamma)

    def test_nan_metric_value(self):
        from custom_test_metric import CMetricFirstNan
        some_nan_metric = CMetricFirstNan()

        print "metric created "

        # Changing default parameters to be sure are not used
        self.svm.set_params({'C': 25, 'kernel.gamma': 1e-1})
        xval_parameters = {'C': [1, 10, 100], 'kernel.gamma': [1, 50]}

        # DO XVAL FOR CHOOSE BEST PARAMETERS
        xval_splitter = CDataSplitter.create(
            'kfold', num_folds=5, random_state=50000)

        # Now we compare the parameters chosen before with a new evaluator
        perf_eval = CPerfEvaluatorXVal(
            xval_splitter, some_nan_metric)
        perf_eval.verbose = 1

        with self.assertRaises(Exception):
            best_params, best_score = perf_eval.evaluate_params(
                self.svm, self.training_dataset, xval_parameters, n_jobs=2,
                pick='last')

            self.logger.info("best score : {:}".format(best_score))

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
            tr, xval_parameters, xval_splitter, 'accuracy', n_jobs=1)

        self.logger.info(
            "Multiclass SVM has now the following parameters: {:}".format(
                multiclass.get_params()))

        for clf_idx, clf in enumerate(multiclass.binary_classifiers):
            self.assertEqual(clf.C, 10.0)
            self.assertEqual(clf.kernel.gamma, 0.1)

        # Final test: train using best parameters
        multiclass.train(tr)

        for clf in multiclass.binary_classifiers:
            for param in best_params:
                self.assertEqual(clf.get_params()[param], best_params[param])

if __name__ == '__main__':
    unittest.main()
