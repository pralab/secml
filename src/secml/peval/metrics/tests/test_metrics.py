import unittest
from secml.utils import CUnitTest

from secml.peval.metrics import CMetric
from secml.array import CArray


class TestCMetrics(CUnitTest):
    """Unittests for CMetric classes."""

    def test_accuracy(self):

        self.logger.info("Testing accuracy score...")
        peval = CMetric.create('accuracy')

        y_true = CArray([0, 1, 2, 3])
        y_pred = CArray([0, 2, 1, 3])

        res = peval.performance_score(y_true=y_true, y_pred=y_pred)
        self.assertEquals(res, 0.5)

        y_true = CArray([0, 1, 0, 0])
        y_pred = CArray([0, 0, 0, 0])

        res = peval.performance_score(y_true=y_true, y_pred=y_pred)
        self.assertEquals(res, 0.75)

    def test_precision(self):

        self.logger.info("Testing precision score...")
        peval = CMetric.create('precision')

        true = CArray([0, 0, 0, 0, 1, 1, 1, 1])
        pred = CArray([1, 0, 0, 0, 1, 1, 0, 0])

        res = peval.performance_score(y_true=true, y_pred=pred)
        # tp: 0.5, fp: 0.25 -> 0.5 / (0.5 + 0.25) = 0.666...
        self.assertAlmostEqual(res, 0.67, 2)

    def test_recall(self):

        self.logger.info("Testing recall score...")
        peval = CMetric.create('recall')

        true = CArray([0, 0, 0, 0, 1, 1, 1, 1])
        pred = CArray([1, 0, 0, 0, 1, 1, 0, 0])

        res = peval.performance_score(y_true=true, y_pred=pred)
        # tp: 0.5, fn: 0.5 -> 0.5 / (0.5 + 0.5) = 0.5
        self.assertEquals(res, 0.5)

    def test_f1(self):

        self.logger.info("Testing f1 score...")
        peval = CMetric.create('f1')

        true = CArray([0, 0, 0, 0, 1, 1, 1, 1])
        pred = CArray([1, 0, 0, 0, 1, 1, 0, 0])

        res = peval.performance_score(y_true=true, y_pred=pred)
        # precision: 0.67, recall: 0.5
        # 2 * (prec * rec) / (prec + rec) -> 2 * 0.335 / 1.17 = 0.57
        self.assertAlmostEqual(res, 0.57, 2)

    def test_mae(self):

        self.logger.info("Testing mae score...")
        peval = CMetric.create('mae')

        true = CArray([3, -0.5, 2, 7])
        pred = CArray([2.5, 0.0, 2, 8])

        res = peval.performance_score(y_true=true, score=pred)
        self.assertEquals(res, 0.5)

    def test_mse(self):

        self.logger.info("Testing mse score...")
        peval = CMetric.create('mse')

        true = CArray([3, -0.5, 2, 7])
        pred = CArray([2.5, 0.0, 2, 8])

        res = peval.performance_score(y_true=true, score=pred)
        self.assertEquals(res, 0.375)

    def test_tpatfp(self):

        self.logger.info("Testing tp_at_fp score...")
        peval = CMetric.create('tp_at_fp', fp_rate=0.1)

        true = CArray([0, 0, 1, 1])
        pred = CArray([0.1, 0.4, 0.35, 0.8])

        res = peval.performance_score(y_true=true, score=pred)
        self.assertEquals(res, 0.5)

    def test_auc(self):

        self.logger.info("Testing auc score...")
        peval = CMetric.create('auc')

        true = CArray([0, 0, 1, 1])
        pred = CArray([0.1, 0.4, 0.35, 0.8])

        res = peval.performance_score(y_true=true, score=pred)
        self.assertEquals(res, 0.75)

        self.logger.info("Testing auc_wmw score...")
        peval = CMetric.create('auc_wmw')

        true = CArray([0, 0, 1, 1])
        pred = CArray([0.1, 0.4, 0.35, 0.8])

        res = peval.performance_score(y_true=true, score=pred)
        self.assertEquals(res, 0.75)

        self.logger.info("Testing pauc score...")
        peval = CMetric.create('pauc', fp_rate=1.0, n_points=500)

        true = CArray([0, 0, 1, 1])
        pred = CArray([0.1, 0.4, 0.35, 0.8])

        res = peval.performance_score(y_true=true, score=pred)
        self.assertEquals(res, 0.75)


if __name__ == '__main__':
    unittest.main()
