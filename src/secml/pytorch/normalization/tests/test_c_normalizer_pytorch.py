from secml.utils import CUnitTest

from secml.pytorch.normalization import CNormalizerPyTorch
from secml.pytorch.classifiers import CClassifierPyTorchMLP
from secml.data.loader import CDLRandom
from secml.optimization import COptimizer
from secml.optimization.function import CFunction


class TestCNormalizerPyTorch(CUnitTest):

    def setUp(self):

        self.ds = CDLRandom(n_samples=100, n_classes=3,
                            n_features=20, n_informative=15,
                            random_state=0).load()

        self.net = CClassifierPyTorchMLP(
            input_dims=20, hidden_dims=(40, ), output_dims=3,
            weight_decay=0, epochs=10, learning_rate=1e-1,
            momentum=0, random_state=0)
        self.net.fit(self.ds)

        self.norm = CNormalizerPyTorch(pytorch_clf=self.net)

    def test_normalization(self):
        """Testing normalization."""
        x = self.ds.X[0, :]

        self.logger.info("Testing normalization at last layer")

        out_norm = self.norm.normalize(x)
        out_net = self.net.get_layer_output(x, layer=None)

        self.logger.info("Output of normalize:\n{:}".format(out_norm))
        self.logger.info("Output of net:\n{:}".format(out_net))

        self.assertFalse((out_norm.round(3) != out_net.round(3)).any())

        self.norm.out_layer = 'linear1'

        self.logger.info(
            "Testing normalization at layer {:}".format(self.norm.out_layer))

        out_norm = self.norm.normalize(x)
        out_net = self.net.get_layer_output(x, layer=self.norm.out_layer)

        self.logger.info("Output of normalize:\n{:}".format(out_norm))
        self.logger.info("Output of net:\n{:}".format(out_net))

        self.assertFalse((out_norm.round(3) != out_net.round(3)).any())

    def test_gradient(self):
        """Test for gradient."""
        x = self.ds.X[0, :]

        layer = None
        self.norm.out_layer = layer
        self.logger.info("Returning gradient for layer: {:}".format(layer))
        grad = self.norm.gradient(x, y=0)

        self.logger.info("Output of gradient_f_x:\n{:}".format(grad))

        self.assertTrue(grad.is_vector_like)
        self.assertEqual(x.size, grad.size)

        layer = 'linear1'
        self.norm.out_layer = layer
        self.logger.info("Returning output for layer: {:}".format(layer))
        out = self.net.get_layer_output(x, layer=layer)
        self.logger.info("Returning gradient for layer: {:}".format(layer))
        grad = self.norm.gradient(x, w=out)

        self.logger.info("Output of gradient_f_x:\n{:}".format(grad))

        self.assertTrue(grad.is_vector_like)
        self.assertEqual(x.size, grad.size)

    def _fun_args(self, x, *args):
        return self.clf.decision_function(x, **args[0])

    def _grad_args(self, x, *args):
        """Wrapper needed as gradient_f_x have **kwargs"""
        return self.clf.gradient_f_x(x, **args[0])

    def test_aspreprocess(self):
        """Test for normalizer used as preprocess."""
        from secml.ml.classifiers import CClassifierSVM
        from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
        self.clf = CClassifierMulticlassOVA(
            classifier=CClassifierSVM, preprocess=self.norm)

        self.logger.info("Testing last layer")

        self.clf.fit(self.ds)

        y_pred, scores = self.clf.predict(
            self.ds.X, return_decision_function=True)
        self.logger.info("TRUE:\n{:}".format(self.ds.Y.tolist()))
        self.logger.info("Predictions:\n{:}".format(y_pred.tolist()))
        self.logger.info("Scores:\n{:}".format(scores))

        x = self.ds.X[0, :]

        self.logger.info("Testing last layer gradient")

        for c in self.ds.classes:

            self.logger.info("Gradient w.r.t. class {:}".format(c))

            grad = self.clf.gradient_f_x(x, y=c)  # y is required for multiclassova

            self.logger.info("Output of gradient_f_x:\n{:}".format(grad))

            check_grad_val = COptimizer(
                CFunction(self._fun_args, self._grad_args)).check_grad(
                    x, ({'y': c}), epsilon=1e-1)
            self.logger.info(
                "norm(grad - num_grad): %s", str(check_grad_val))
            self.assertLess(check_grad_val, 1e-3)

            self.assertTrue(grad.is_vector_like)
            self.assertEqual(x.size, grad.size)

        layer = 'linear1'
        self.norm.out_layer = layer

        self.logger.info("Testing layer {:}".format(self.norm.out_layer))

        self.clf.fit(self.ds)

        y_pred, scores = self.clf.predict(
            self.ds.X, return_decision_function=True)
        self.logger.info("TRUE:\n{:}".format(self.ds.Y.tolist()))
        self.logger.info("Predictions:\n{:}".format(y_pred.tolist()))
        self.logger.info("Scores:\n{:}".format(scores))

        self.logger.info("Testing 'linear1' layer gradient")
        grad = self.clf.gradient_f_x(x, y=0)  # y is required for multiclassova
        self.logger.info("Output of gradient_f_x:\n{:}".format(grad))

        self.assertTrue(grad.is_vector_like)
        self.assertEqual(x.size, grad.size)


if __name__ == '__main__':
    CUnitTest.main()
