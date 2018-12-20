from secml.utils import CUnitTest

from secml.ml.classifiers.loss import CLossCrossEntropy, softmax
from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.array import CArray


class TestCLossCrossEntropy(CUnitTest):
    """Unittests for CLossCrossEntropy and softmax."""

    def setUp(self):

        self.ds = CDLRandom(n_classes=3, n_samples=50, random_state=0,
                            n_informative=3).load()

        self.logger.info("Fit an SVM and classify dataset...")
        self.ova = CClassifierMulticlassOVA(CClassifierSVM)
        self.ova.fit(self.ds)
        self.labels, self.scores = self.ova.predict(self.ds.X)

    def test_softmax(self):
        """Unittests for softmax function."""
        from sklearn.utils.extmath import softmax as softmax_sk

        sm = softmax(self.scores)
        sm_sk = softmax_sk(self.scores.tondarray())

        self.logger.info("Our softmax.max():\n{:}".format(sm.max()))
        self.logger.info("SKlearn softmax.max():\n{:}".format(sm_sk.max()))

        self.assertFalse((sm.round(4) != CArray(sm_sk).round(4)).any())

        self.logger.info("Testing a single point...")

        sm = softmax(self.scores[0, :])
        sm_sk = softmax_sk(self.scores[0, :].tondarray())

        self.logger.info("Our softmax.max():\n{:}".format(sm.max()))
        self.logger.info("SKlearn softmax.max():\n{:}".format(sm_sk.max()))

        self.assertFalse((sm.round(4) != CArray(sm_sk).round(4)).any())

    def test_in_out(self):
        """Unittest for input and output to CCrossEntropy"""
        def _check_loss(l, n_samples):

            self.assertIsInstance(l, CArray)
            self.assertTrue(l.isdense)
            self.assertEqual(1, l.ndim)
            self.assertEqual(n_samples, l.size)
            self.assertEqual(l.dtype, float)

        loss_class = CLossCrossEntropy()

        loss = loss_class.loss(self.ds.Y, self.scores)
        loss_mean = loss.mean()
        self.logger.info(
            "{:}.loss(y_true, scores).mean():\n{:}".format(
                loss_class.__class__.__name__, loss_mean))
        _check_loss(loss, self.ds.Y.size)

        loss = loss_class.loss(self.ds.Y, self.scores, pos_label=0)
        loss_mean = loss.mean()
        self.logger.info(
            "{:}.loss(y_true, scores, pos_label=0).mean():\n{:}".format(
                loss_class.__class__.__name__, loss_mean))
        _check_loss(loss, self.ds.Y.size)

        loss = loss_class.loss(self.ds.Y, self.scores, pos_label=2)
        loss_mean = loss.mean()
        self.logger.info(
            "{:}.loss(y_true, scores, pos_label=0).mean():\n{:}".format(
                loss_class.__class__.__name__, loss_mean))
        _check_loss(loss, self.ds.Y.size)

        loss = loss_class.loss(self.ds.Y[0], self.scores[0, :])
        loss_mean = loss.mean()
        self.logger.info(
            "{:}.loss(y_true[0], scores[0,:]).mean():\n{:}".format(
                loss_class.__class__.__name__, loss_mean))
        _check_loss(loss, 1)


if __name__ == '__main__':
    CUnitTest.main()
