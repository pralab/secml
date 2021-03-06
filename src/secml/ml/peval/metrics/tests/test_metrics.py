from secml.testing import CUnitTest

from secml.ml.peval.metrics import CMetric
from secml.array import CArray
from secml.core.type_utils import is_float


class TestCMetrics(CUnitTest):
    """Unittests for CMetric classes."""

    def test_accuracy(self):

        self.logger.info("Testing accuracy metric...")
        peval = CMetric.create('accuracy')

        y_true = CArray([0, 1, 2, 3])
        y_pred = CArray([0, 2, 1, 3])

        res = peval.performance_score(y_true=y_true, y_pred=y_pred)
        self.assertEqual(0.5, res)

        y_true = CArray([0, 1, 0, 0])
        y_pred = CArray([0, 0, 0, 0])

        res = peval.performance_score(y_true=y_true, y_pred=y_pred)
        self.assertEqual(0.75, res)
        self.assertTrue(is_float(res))

    def test_precision(self):

        self.logger.info("Testing precision metric...")
        peval = CMetric.create('precision')

        true = CArray([0, 0, 0, 0, 1, 1, 1, 1])
        pred = CArray([1, 0, 0, 0, 1, 1, 0, 0])

        res = peval.performance_score(y_true=true, y_pred=pred)
        # tpr: 0.5, fpr: 0.25 -> 0.5 / (0.5 + 0.25) = 0.666...
        self.assertAlmostEqual(res, 0.67, 2)
        self.assertTrue(is_float(res))

    def test_recall(self):

        self.logger.info("Testing recall metric...")
        peval = CMetric.create('recall')

        true = CArray([0, 0, 0, 0, 1, 1, 1, 1])
        pred = CArray([1, 0, 0, 0, 1, 1, 0, 0])

        res = peval.performance_score(y_true=true, y_pred=pred)
        # tpr: 0.5, fnr: 0.5 -> 0.5 / (0.5 + 0.5) = 0.5
        self.assertEqual(0.5, res)
        self.assertTrue(is_float(res))

    def test_f1(self):

        self.logger.info("Testing F1 score metric...")
        peval = CMetric.create('f1')

        true = CArray([0, 0, 0, 0, 1, 1, 1, 1])
        pred = CArray([1, 0, 0, 0, 1, 1, 0, 0])

        res = peval.performance_score(y_true=true, y_pred=pred)
        # precision: 0.67, recall: 0.5
        # 2 * (prec * rec) / (prec + rec) -> 2 * 0.335 / 1.17 = 0.57
        self.assertAlmostEqual(res, 0.57, 2)
        self.assertTrue(is_float(res))

    def test_mae(self):

        self.logger.info("Testing MAE metric...")
        peval = CMetric.create('mae')

        true = CArray([3, -0.5, 2, 7])
        pred = CArray([2.5, 0.0, 2, 8])

        res = peval.performance_score(y_true=true, score=pred)
        self.assertEqual(0.5, res)
        self.assertTrue(is_float(res))

    def test_mse(self):

        self.logger.info("Testing MSE metric...")
        peval = CMetric.create('mse')

        true = CArray([3, -0.5, 2, 7])
        pred = CArray([2.5, 0.0, 2, 8])

        res = peval.performance_score(y_true=true, score=pred)
        self.assertEqual(0.375, res)
        self.assertTrue(is_float(res))

    def _test_roc_metric(self, metric):
        """Test for ROC-related metrics which get y_true and score_pred.

        Parameters
        ----------
        metric : CMetric

        Returns
        -------
        res : float

        """
        true = CArray([0, 0, 1, 0, 1, 1])
        pred = CArray([0.1, 0.2, 0.3, 0.4, 0.75, 0.8])

        res = metric.performance_score(y_true=true, score=pred)
        self.assertTrue(is_float(res))

        return res

    def test_tpratfpr(self):

        self.logger.info("Testing TPR @ FPR metric...")
        metric = CMetric.create('tpr-at-fpr', fpr=0.1)

        res = self._test_roc_metric(metric)

        self.assertAlmostEqual(0.67, res, places=2)

    def test_fnratfpr(self):

        self.logger.info("Testing FNR @ FPR metric...")
        metric = CMetric.create('fnr-at-fpr', fpr=0.1)

        res = self._test_roc_metric(metric)

        self.assertAlmostEqual(0.33, res, places=2)

    def test_thatfpr(self):

        self.logger.info("Testing TH @ FPR metric...")
        metric = CMetric.create('th-at-fpr', fpr=0.1)

        res = self._test_roc_metric(metric)

        self.assertEqual(0.645, res)

    def test_tpratth(self):

        self.logger.info("Testing TPR @ TH metric...")
        metric = CMetric.create('tpr-at-th', th=0.76)

        res = self._test_roc_metric(metric)

        self.assertAlmostEqual(0.33, res, places=2)

    def test_fnratth(self):

        self.logger.info("Testing FNR @ TH metric...")
        metric = CMetric.create('fnr-at-th', th=0.76)

        res = self._test_roc_metric(metric)

        self.assertAlmostEqual(0.67, res, places=2)

    def test_auc(self):

        self.logger.info("Testing AUC metric...")
        metric = CMetric.create('auc')

        res = self._test_roc_metric(metric)

        self.assertAlmostEqual(0.89, res, places=2)

        self.logger.info("Testing AUC-WMW metric...")
        metric = CMetric.create('auc-wmw')

        res = self._test_roc_metric(metric)

        self.assertAlmostEqual(0.89, res, places=2)

        self.logger.info("Testing pAUC metric...")
        metric = CMetric.create('pauc', fpr=1.0, n_points=500)

        res = self._test_roc_metric(metric)

        self.assertAlmostEqual(0.89, res, places=2)


if __name__ == '__main__':
    CUnitTest.main()
