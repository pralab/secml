from secml.testing import CUnitTest

import sklearn.metrics as skm
import numpy as np

from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.kernels import CKernel
from secml.data.splitter import CDataSplitter
from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.ml.peval import CPerfEvaluatorXVal
from secml.ml.peval.metrics import CMetric
from secml.core.constants import nan


class CMetricFirstNan(CMetric):
    """Test metric which returns some nans."""
    best_value = 1.0

    def __init__(self):
        self._count = 0

    def _performance_score(self, y_true, score):
        if self._count == 0:
            self._count += 1
            return nan
        else:
            return 1


class CMetricAllNan(CMetric):
    """Test metric which returns all nans."""
    best_value = 1.0

    def __init__(self):
        pass

    def _performance_score(self, y_true, score):
        return nan


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
        for comb in range(len(parameters_combination)):

            this_fold_score = []
            num_xval_fold = len(xval_splitter.tr_idx)

            for f in range(num_xval_fold):

                self.svm.set("C", parameters_combination[comb][0])
                self.svm.kernel.gamma = parameters_combination[comb][1]

                self.svm.fit(
                    self.training_dataset[xval_splitter.tr_idx[f], :])

                this_fold_predicted = self.svm.predict(
                    self.training_dataset[xval_splitter.ts_idx[f], :].X)

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

        # Changing default parameters to be sure are not used
        self.svm.set_params({'C': 25, 'kernel.gamma': 1e-1})
        xval_parameters = {'C': [1, 10, 100], 'kernel.gamma': [1, 50]}

        # DO XVAL FOR CHOOSE BEST PARAMETERS
        xval_splitter = CDataSplitter.create(
            'kfold', num_folds=5, random_state=50000)

        self.logger.info("Testing metric with some nan")

        some_nan_metric = CMetricFirstNan()

        # Now we compare the parameters chosen before with a new evaluator
        perf_eval = CPerfEvaluatorXVal(
            xval_splitter, some_nan_metric)
        perf_eval.verbose = 1

        best_params, best_score = perf_eval.evaluate_params(
            self.svm, self.training_dataset, xval_parameters, pick='last')

        self.logger.info("best score : {:}".format(best_score))

        # The xval should select the only one actual value (others are nan)
        self.assertEqual(best_score, 1.)

        self.logger.info("Testing metric with all nan")

        # This test case involves an all-nan slice
        self.logger.filterwarnings(
            action="ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning
        )

        all_nan_metric = CMetricAllNan()

        # Now we compare the parameters chosen before with a new evaluator
        perf_eval = CPerfEvaluatorXVal(
            xval_splitter, all_nan_metric)
        perf_eval.verbose = 1

        with self.assertRaises(ValueError):
            perf_eval.evaluate_params(
                self.svm, self.training_dataset, xval_parameters, pick='last')

    def _run_multiclass(self, tr, multiclass, xval_params, expected_best):

        xval_splitter = CDataSplitter.create(
            'kfold', num_folds=3, random_state=50000)

        # Set the best parameters inside the classifier
        best_params = multiclass.estimate_parameters(
            tr, xval_params, xval_splitter, 'accuracy', n_jobs=1)

        self.logger.info(
            "Multiclass SVM has now the following parameters: {:}".format(
                multiclass.get_params()))

        for clf_idx, clf in enumerate(multiclass._binary_classifiers):
            self.assertEqual(clf.C, expected_best['C'])
            self.assertEqual(clf.kernel.gamma, expected_best['kernel.gamma'])

        # Final test: fit using best parameters
        multiclass.fit(tr)

        for clf in multiclass._binary_classifiers:
            for param in best_params:
                self.assertEqual(clf.get_params()[param], best_params[param])

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

        expected = {'C': 10.0, 'kernel.gamma': 0.1}

        self._run_multiclass(tr, multiclass, xval_parameters, expected)

        self.logger.info("Testing with preprocessor")

        kernel = CKernel.create('rbf')
        multiclass = CClassifierMulticlassOVA(
            CClassifierSVM, C=1, kernel=kernel, preprocess='min-max')
        multiclass.verbose = 1

        xval_parameters = {'C': [1, 10, 100], 'kernel.gamma': [0.1, 1]}

        expected = {'C': 10.0, 'kernel.gamma': 0.1}

        self._run_multiclass(tr, multiclass, xval_parameters, expected)


if __name__ == '__main__':
    CUnitTest.main()
