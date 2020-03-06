from secml.testing import CUnitTest

from secml.array import CArray
from secml.data.loader import CDataLoaderMNIST
from secml.data.splitter import CDataSplitterKFold
from secml.ml.peval.metrics import CMetricAccuracy
from secml.ml.classifiers import \
    CClassifierSVM, CClassifierLogistic, CClassifierRidge
from secml.ml.kernels import CKernelRBF
from secml.ml.classifiers.gradients.tests.test_classes import \
    CClassifierGradientTest

from secml.explanation import CExplainerInfluenceFunctions


class TestCExplainerInfluenceFunctions(CUnitTest):
    """Unittests for CExplainerInfluenceFunctions."""

    @classmethod
    def setUpClass(cls):
        CUnitTest.setUpClass()

        cls._tr, cls._val, cls._ts = cls._create_mnist_dataset()
        cls._metric = CMetricAccuracy()

    def test_explanation_svm(self):
        self._clf = CClassifierSVM()
        self._clf.store_dual_vars = True
        self._clf_idx = 'lin-svm'

        self._test_explanation_simple_clf()

    def test_explanation_logistic(self):
        self._clf = CClassifierLogistic()
        self._clf_idx = 'logistic regression'

        self._test_explanation_simple_clf()

    def test_explanation_svm_rbf(self):
        self._clf = CClassifierSVM(kernel=CKernelRBF(gamma=0.01), C=10)
        self._clf.kernel.gamma = 0.01
        self._clf.store_dual_vars = True
        self._clf_idx = 'rbf-svm'

        self._test_explanation_simple_clf()

    def test_explanation_ridge(self):
        self._clf = CClassifierRidge()
        self._clf_idx = 'Ridge'

        self._test_explanation_simple_clf()

    @staticmethod
    def _create_mnist_dataset(
            digits=[4, 9], n_tr=100, n_val=200, n_ts=200, seed=4):  # 10
        loader = CDataLoaderMNIST()

        tr = loader.load('training', digits=digits)
        ts = loader.load('testing', digits=digits, num_samples=n_ts)

        # start train and validation dataset split
        splitter = CDataSplitterKFold(num_folds=2, random_state=seed)
        splitter.compute_indices(tr)

        val_dts_idx = CArray.randsample(CArray.arange(0, tr.num_samples),
                                        n_val, random_state=seed)
        val = tr[val_dts_idx, :]

        tr_dts_idx = CArray.randsample(CArray.arange(0, tr.num_samples),
                                       n_tr, random_state=seed)
        tr = tr[tr_dts_idx, :]

        tr.X /= 255.0
        val.X /= 255.0
        ts.X /= 255.0

        return tr, val, ts

    def _check_accuracy(self):
        preds = self._clf.predict(self._ts.X)
        acc = self._metric.performance_score(y_true=self._ts.Y, y_pred=preds)
        self.logger.info("Classifier accuracy: {:} ".format(acc))
        self.assertGreater(acc, 0.70)

    def _compute_influences(self):
        self._clf_loss = self._clf._loss.class_type

        self._clf.fit(self._tr)

        self._check_accuracy()

        explanation = CExplainerInfluenceFunctions(self._clf, self._tr,
                                                   outer_loss_idx=self._clf_loss)
        self.influences = explanation.explain(self._ts.X, self._ts.Y)
        self.clf_gradients = CClassifierGradientTest.create(
            self._clf.class_type)

    def _get_tr_without_point(self, p_idx):
        """
        Given the idx of a point return a copy of the training dataset
        without that point

        Parameters
        ----------
        p_idx int
            idx of the point that is wanted to be excluded by the training
            dataset

        Returns
        -------
        new_tr CDataset
                dataset without the point with the given index
        """
        all_idx = CArray.arange(self._tr.num_samples)
        not_p_idx = all_idx.find(all_idx != p_idx)
        new_tr = self._tr[not_p_idx, :]
        return new_tr

    def _check_influence(self, point_idx):
        """
        This function learn the classifier without a point and return the
        classifier loss on the test set.

        Parameters
        ----------
        point CArray
            training sample
        """
        clf_copy = self._clf.deepcopy()
        new_dataset = self._get_tr_without_point(point_idx)

        clf_copy.fit(new_dataset)

        loss = (1 / self._ts.num_samples) * self.clf_gradients.l(
            self._ts.X, self._ts.Y, clf_copy).sum(axis=None)

        return loss

    def _check_prototype_pair(self, p_inf_idx, p_not_inf_idx):
        """
        Given a pair of prototypes where one is supposed to be between the
        most influent and the one between the less influent check if this is true.

        Parameters
        ----------
        p_inf_idx integer
            index in the training set of a sample supposed to be between the
            most influent
        p_not_inf_idx integer
            index in the training set of a sample supposed to be between the
            less influent
        """
        acc_without_p_infl = self._check_influence(p_inf_idx)
        self.logger.info("The loss without the point {:} supposed to be "
                         "one of the most influent is {:}".format(p_inf_idx,
                                                                  acc_without_p_infl))
        acc_without_p_not_infl = self._check_influence(p_not_inf_idx)
        self.logger.info("The loss without the point {:} supposed to be "
                         "one of the less influent is {:}".format(
            p_not_inf_idx,
            acc_without_p_not_infl))

        self.assertGreater(acc_without_p_infl, acc_without_p_not_infl,
                           "The point that is supposed to be between the "
                           "less influent has a higher influence of the "
                           "point supposed to be between one of the most "
                           "influent")

    def _test_explanation(self):
        self._compute_influences()

        self.assertEqual(self.influences.shape,
                         (self._ts.num_samples, self._tr.num_samples),
                         "The shape of the influences is wrong!")

        average_influence = self.influences.mean(axis=0).ravel()
        # order the idx of the tr samples in the way to have the less
        # influent in the first position of the vector and the most influent
        # in the las ones
        avg_infl_idx = average_influence.argsort()

        n_check = 2
        for i in range(1, n_check + 1):
            not_infl_idx = avg_infl_idx[i - 1].item()
            infl_idx = avg_infl_idx[-i].item()
            self._check_prototype_pair(infl_idx, not_infl_idx)

    def _test_explanation_simple_clf(self):
        self.logger.info("Explain the decisions of a {:} classifier and "
                         "test if they are reasonable".format(self._clf_idx))
        self._test_explanation()


if __name__ == '__main__':
    CUnitTest.main()
