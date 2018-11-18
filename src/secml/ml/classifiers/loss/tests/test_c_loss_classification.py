from secml.utils import CUnitTest

from secml.ml.classifiers.loss import *
from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierSVM
from secml.array import CArray
from secml.figure import CFigure


class TestCLossClassification(CUnitTest):
    """Unittests for CLossClassification and subclasses."""

    def setUp(self):

        self.ds = CDLRandom(n_samples=50, random_state=0).load()

        self.logger.info("Train an SVM and classify dataset...")
        self.svm = CClassifierSVM()
        self.svm.train(self.ds)
        self.labels, self.scores = self.svm.classify(self.ds.X)

    def test_one_at_zero(self):
        """Testing that classification loss return 1 for input 0."""

        for loss_id in ('hinge', 'hinge_squared', 'square', 'logistic'):

            self.logger.info("Creating loss: {:}".format(loss_id))
            loss_class = CLoss.create(loss_id)

            self.assertEqual(
                CArray([1.0]), loss_class.loss(CArray([1]), CArray([0])))

    def test_in_out(self):
        """Unittest for input and output to loss classes"""
        def _check_loss(l, n_samples):

            self.assertIsInstance(l, CArray)
            self.assertTrue(l.isdense)
            self.assertEqual(1, l.ndim)
            self.assertEqual(n_samples, l.size)
            self.assertEqual(l.dtype, float)

        for loss_id in ('hinge', 'hinge_squared', 'square', 'logistic'):

            self.logger.info("Creating loss: {:}".format(loss_id))
            loss_class = CLoss.create(loss_id)

            loss = loss_class.loss(self.ds.Y, self.scores)
            loss_mean = loss.mean()
            self.logger.info(
                "{:}.loss(y_true, scores).mean():\n{:}".format(
                    loss_class.__class__.__name__, loss_mean))
            _check_loss(loss, self.ds.Y.size)

            loss_pos = loss_class.loss(self.ds.Y, self.scores[:, 1].ravel())
            loss_mean_pos = loss_pos.mean()
            self.logger.info(
                "{:}.loss(y_true, scores[:, 1].ravel()).mean():\n".format(
                    loss_class.__class__.__name__, loss_mean_pos))
            _check_loss(loss_pos, self.ds.Y.size)

            self.assertEqual(loss_mean, loss_mean_pos)

            loss = loss_class.loss(self.ds.Y, self.scores, pos_label=0)
            loss_mean = loss.mean()
            self.logger.info(
                "{:}.loss(y_true, scores, pos_label=0).mean():\n{:}".format(
                    loss_class.__class__.__name__, loss_mean))
            _check_loss(loss, self.ds.Y.size)

            loss_neg = loss_class.loss(self.ds.Y, self.scores[:, 0].ravel())
            loss_mean_neg = loss_neg.mean()
            self.logger.info(
                "{:}.loss(y_true, scores[:,0].ravel()).mean():\n".format(
                    loss_class.__class__.__name__, loss_mean_neg))
            _check_loss(loss_neg, self.ds.Y.size)

            self.assertEqual(loss_mean, loss_mean_neg)

            loss = loss_class.loss(self.ds.Y[0], self.scores[0, :])
            loss_mean = loss.mean()
            self.logger.info(
                "{:}.loss(y_true[0], scores[0,:]).mean():\n{:}".format(
                    loss_class.__class__.__name__, loss_mean))
            _check_loss(loss, 1)

    def test_draw(self):
        """Drawing the loss functions.

        Inspired by: https://en.wikipedia.org/wiki/Loss_functions_for_classification

        """
        fig = CFigure()
        x = CArray.arange(-1, 3.01, 0.01)

        fig.sp.plot(x, CArray([1 if i <= 0 else 0 for i in x]),
                    label='0-1 indicator')

        for loss_id in ('hinge', 'hinge_squared', 'square', 'logistic'):

            self.logger.info("Creating loss: {:}".format(loss_id))
            loss_class = CLoss.create(loss_id)
            fig.sp.plot(x, loss_class.loss(CArray([1]), x), label=loss_id)

        fig.sp.grid()
        fig.sp.legend()

        fig.show()


if __name__ == '__main__':
    CUnitTest.main()
