from secml.testing import CUnitTest

from secml.ml.classifiers.loss import *
from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierSVM
from secml.array import CArray
from secml.figure import CFigure


class TestCLossRegression(CUnitTest):
    """Unittests for CLossRegression and subclasses."""

    def setUp(self):

        self.ds = CDLRandom(n_samples=50, random_state=0).load()

        self.logger.info("Train an SVM and classify dataset...")
        self.svm = CClassifierSVM()
        self.svm.fit(self.ds)
        self.labels, self.scores = self.svm.predict(
            self.ds.X, return_decision_function=True)

    def test_in_out(self):
        """Unittest for input and output to loss classes"""
        def _check_loss(l, n_samples):

            self.assertIsInstance(l, CArray)
            self.assertTrue(l.isdense)
            self.assertEqual(1, l.ndim)
            self.assertEqual(n_samples, l.size)
            self.assertEqual(l.dtype, float)

        for loss_id in ('e-insensitive',
                        'e-insensitive-squared',
                        'quadratic'):

            self.logger.info("Creating loss: {:}".format(loss_id))
            loss_class = CLoss.create(loss_id)

            loss_pos = loss_class.loss(self.ds.Y, self.scores[:, 1].ravel())
            loss_mean_pos = loss_pos.mean()
            self.logger.info(
                "{:}.loss(y_true, scores[:, 1].ravel()).mean():\n".format(
                    loss_class.__class__.__name__, loss_mean_pos))
            _check_loss(loss_pos, self.ds.Y.size)

            loss = loss_class.loss(self.ds.Y[0], self.scores[0, 1].ravel())
            loss_mean = loss.mean()
            self.logger.info(
                "{:}.loss(y_true[0], scores[0,:]).mean():\n{:}".format(
                    loss_class.__class__.__name__, loss_mean))
            _check_loss(loss, 1)

            with self.assertRaises(ValueError):
                loss_class.loss(self.ds.Y, self.scores[:, 1])

    def test_draw(self):
        """Drawing the loss functions.

        Inspired by: https://en.wikipedia.org/wiki/Loss_functions_for_classification

        """
        fig = CFigure()
        x = CArray.arange(-1, 3.01, 0.01)

        for loss_id in ('e-insensitive',
                        'e-insensitive-squared',
                        'quadratic'):

            self.logger.info("Creating loss: {:}".format(loss_id))
            loss_class = CLoss.create(loss_id)
            fig.sp.plot(x, loss_class.loss(CArray([1]), x), label=loss_id)

        fig.sp.grid()
        fig.sp.legend()

        fig.show()


if __name__ == '__main__':
    CUnitTest.main()
