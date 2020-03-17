from secml.testing import CUnitTest

from secml.ml.classifiers.loss import *
from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierSVM
from secml.array import CArray
from secml.figure import CFigure
from secml.optim.function import CFunction


class TestCLossClassification(CUnitTest):
    """Unittests for CLossClassification and subclasses."""

    def setUp(self):

        self.ds = CDLRandom(n_samples=50, random_state=0).load()

        self.logger.info("Train an SVM and classify dataset...")
        self.svm = CClassifierSVM()
        self.svm.fit(self.ds)
        self.labels, self.scores = self.svm.predict(
            self.ds.X, return_decision_function=True)

    def test_one_at_zero(self):
        """Testing that classification loss return 1 for input 0."""

        for loss_id in ('hinge', 'hinge-squared', 'square', 'log'):

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

        for loss_id in ('hinge', 'hinge-squared', 'square', 'log'):

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

        for loss_id in ('hinge', 'hinge-squared', 'square', 'log'):

            self.logger.info("Creating loss: {:}".format(loss_id))
            loss_class = CLoss.create(loss_id)
            fig.sp.plot(x, loss_class.loss(CArray([1]), x), label=loss_id)

        fig.sp.grid()
        fig.sp.legend()

        fig.show()

    def test_grad(self):
        """Compare analytical gradients with its numerical approximation."""
        def _loss_wrapper(scores, loss, true_labels):
            return loss.loss(true_labels, scores)

        def _dloss_wrapper(scores, loss, true_labels):
            return loss.dloss(true_labels, scores)

        for loss_id in ('hinge', 'hinge-squared', 'square', 'log'):
            self.logger.info("Creating loss: {:}".format(loss_id))
            loss_class = CLoss.create(loss_id)

            n_elemes = 1
            y_true = CArray.randint(0, 2, n_elemes).todense()
            score = CArray.randn((n_elemes,))

            check_grad_val = CFunction(
                _loss_wrapper, _dloss_wrapper).check_grad(
                score, 1e-8, loss=loss_class, true_labels=y_true)
            self.logger.info("Gradient difference between analytical svm "
                             "gradient and numerical gradient: %s",
                             str(check_grad_val))
            self.assertLess(check_grad_val, 1e-4,
                            "the gradient is wrong {:} for {:} loss".format(
                                check_grad_val, loss_id))


if __name__ == '__main__':
    CUnitTest.main()
