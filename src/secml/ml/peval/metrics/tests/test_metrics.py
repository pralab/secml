from secml.testing import CUnitTest

from secml.ml.peval.metrics import CMetric
from secml.array import CArray
from secml.core.type_utils import is_float


class TestCMetrics(CUnitTest):
    """Unittests for CMetric classes."""

    def test_accuracy(self):

        self.logger.info("Testing accuracy score...")
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

        self.logger.info("Testing precision score...")
        peval = CMetric.create('precision')

        true = CArray([0, 0, 0, 0, 1, 1, 1, 1])
        pred = CArray([1, 0, 0, 0, 1, 1, 0, 0])

        res = peval.performance_score(y_true=true, y_pred=pred)
        # tpr: 0.5, fpr: 0.25 -> 0.5 / (0.5 + 0.25) = 0.666...
        self.assertAlmostEqual(res, 0.67, 2)
        self.assertTrue(is_float(res))

    def test_recall(self):

        self.logger.info("Testing recall score...")
        peval = CMetric.create('recall')

        true = CArray([0, 0, 0, 0, 1, 1, 1, 1])
        pred = CArray([1, 0, 0, 0, 1, 1, 0, 0])

        res = peval.performance_score(y_true=true, y_pred=pred)
        # tpr: 0.5, fnr: 0.5 -> 0.5 / (0.5 + 0.5) = 0.5
        self.assertEqual(0.5, res)
        self.assertTrue(is_float(res))

    def test_f1(self):

        self.logger.info("Testing f1 score...")
        peval = CMetric.create('f1')

        true = CArray([0, 0, 0, 0, 1, 1, 1, 1])
        pred = CArray([1, 0, 0, 0, 1, 1, 0, 0])

        res = peval.performance_score(y_true=true, y_pred=pred)
        # precision: 0.67, recall: 0.5
        # 2 * (prec * rec) / (prec + rec) -> 2 * 0.335 / 1.17 = 0.57
        self.assertAlmostEqual(res, 0.57, 2)
        self.assertTrue(is_float(res))

    def test_mae(self):

        self.logger.info("Testing mae score...")
        peval = CMetric.create('mae')

        true = CArray([3, -0.5, 2, 7])
        pred = CArray([2.5, 0.0, 2, 8])

        res = peval.performance_score(y_true=true, score=pred)
        self.assertEqual(0.5, res)
        self.assertTrue(is_float(res))

    def test_mse(self):

        self.logger.info("Testing mse score...")
        peval = CMetric.create('mse')

        true = CArray([3, -0.5, 2, 7])
        pred = CArray([2.5, 0.0, 2, 8])

        res = peval.performance_score(y_true=true, score=pred)
        self.assertEqual(0.375, res)
        self.assertTrue(is_float(res))

    def test_tpratfpr(self):

        self.logger.info("Testing tpr_at_fpr score...")
        peval = CMetric.create('tpr-at-fpr', fpr=0.1)

        true = CArray([0, 0, 1, 1])
        pred = CArray([0.1, 0.4, 0.35, 0.8])

        res = peval.performance_score(y_true=true, score=pred)
        self.assertEqual(0.5, res)
        self.assertTrue(is_float(res))

    def test_auc(self):

        self.logger.info("Testing auc score...")
        peval = CMetric.create('auc')

        true = CArray([0, 0, 1, 1])
        pred = CArray([0.1, 0.4, 0.35, 0.8])

        res = peval.performance_score(y_true=true, score=pred)
        self.assertEqual(0.75, res)
        self.assertTrue(is_float(res))

        self.logger.info("Testing auc_wmw score...")
        peval = CMetric.create('auc-wmw')

        true = CArray([0, 0, 1, 1])
        pred = CArray([0.1, 0.4, 0.35, 0.8])

        res = peval.performance_score(y_true=true, score=pred)
        self.assertEqual(0.75, res)
        self.assertTrue(is_float(res))

        self.logger.info("Testing pauc score...")
        peval = CMetric.create('pauc', fpr=1.0, n_points=500)

        true = CArray([0, 0, 1, 1])
        pred = CArray([0.1, 0.4, 0.35, 0.8])

        res = peval.performance_score(y_true=true, score=pred)
        self.assertEqual(0.75, res)
        self.assertTrue(is_float(res))


if __name__ == '__main__':
    CUnitTest.main()
