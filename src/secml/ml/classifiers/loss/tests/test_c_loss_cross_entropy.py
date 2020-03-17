from secml.testing import CUnitTest

from secml.ml.classifiers.loss import CLossCrossEntropy
from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.array import CArray
from secml.optim.function import CFunction
from secml.core.constants import eps


class TestCLossCrossEntropy(CUnitTest):
    """Unittests for CLossCrossEntropy and softmax."""

    def setUp(self):
        self.ds = CDLRandom(n_classes=3, n_samples=50, random_state=0,
                            n_informative=3).load()

        self.logger.info("Fit an SVM and classify dataset...")
        self.ova = CClassifierMulticlassOVA(CClassifierSVM)
        self.ova.fit(self.ds)
        self.labels, self.scores = self.ova.predict(
            self.ds.X, return_decision_function=True)

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

        loss = loss_class.loss(self.ds.Y[0], self.scores[0, :])
        loss_mean = loss.mean()
        self.logger.info(
            "{:}.loss(y_true[0], scores[0,:]).mean():\n{:}".format(
                loss_class.__class__.__name__, loss_mean))
        _check_loss(loss, 1)

    def test_grad(self):
        """Compare analytical gradients with its numerical approximation."""
        def _loss_wrapper(scores, loss, true_labels):
            return loss.loss(true_labels, scores)

        loss_class = CLossCrossEntropy()

        y_true = CArray.randint(0, 2, 1)
        score = CArray.randn((1, 3))

        self.logger.info("Y_TRUE: {:} SCORES: {:}".format(y_true, score))

        for pos_label in (None, 0, 1, 2):
            self.logger.info("POS_LABEL: {:}".format(pos_label))

            # real value of the gradient on x
            grad = loss_class.dloss(y_true, score, pos_label)

            self.logger.info("GRAD: {:}".format(grad))

            approx = CFunction(_loss_wrapper).approx_fprime(
                score, eps, loss_class, y_true)
            self.logger.info("APPROX (FULL): {:}".format(approx))

            pos_label = pos_label if pos_label is not None else y_true.item()
            approx = approx[pos_label]

            self.logger.info("APPROX (POS_LABEL): {:}".format(approx))

            check_grad_val = (grad - approx).norm()

            self.logger.info("Gradient difference between analytical svm "
                             "gradient and numerical gradient: %s",
                             str(check_grad_val))
            self.assertLess(check_grad_val, 1e-4,
                            "the gradient is wrong {:}".format(check_grad_val))


if __name__ == '__main__':
    CUnitTest.main()
