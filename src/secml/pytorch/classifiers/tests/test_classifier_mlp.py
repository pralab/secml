from secml.utils import CUnitTest

from secml.pytorch.classifiers import CClassifierPyTorchMLP
from secml.data.loader import CDLRandom
from secml.ml.peval.metrics import CMetricAccuracy
from secml.optimization import COptimizer
from secml.optimization.function import CFunction


class TestCClassifierPyTorchMLP(CUnitTest):

    def setUp(self):

        self.ds = CDLRandom(n_samples=100, n_classes=3,
                            n_features=20, n_informative=15,
                            random_state=0).load()

        self.clf = CClassifierPyTorchMLP(
            input_dims=20, hidden_dims=(40, ), output_dims=3,
            weight_decay=0, epochs=10, learning_rate=1e-1,
            momentum=0, random_state=0)
        self.clf.verbose = 2

    def test_model_creation(self):
        """Testing model creation with different number of layers/neurons."""
        def check_layers(in_dims, h_dims, out_dims):
            self.logger.info("Testing dims: in {:}, hid {:}, out {:}".format(
                in_dims, h_dims, out_dims))

            nn = CClassifierPyTorchMLP(in_dims, h_dims, out_dims)

            layers = nn._model._modules.items()

            # Expected number of layers: 4 if hl=1, 6 if hl=2, 8 if hl=3, etc.
            self.assertEqual(2 * (len(h_dims) - 1) + 3, len(layers))

            self.assertEqual('linear1', layers[0][0])
            self.assertEqual(in_dims, layers[0][1].in_features)
            self.assertEqual(h_dims[0], layers[0][1].out_features)

            for hl_i, hl_dims in enumerate(h_dims[1:]):
                hl_idx = hl_i + 1  # Index of the hidden layer in layers list
                self.assertEqual('linear' + str(hl_i + 2),
                                 layers[hl_idx * 2][0])
                self.assertEqual('relu' + str(hl_i + 2),
                                 layers[(hl_idx * 2) + 1][0])

                self.assertEqual(h_dims[hl_i],
                                 layers[hl_idx * 2][1].in_features)
                self.assertEqual(h_dims[hl_i + 1],
                                 layers[hl_idx * 2][1].out_features)

            self.assertEqual('linear' + str(len(h_dims) + 1),
                             layers[-1][0])
            self.assertEqual(h_dims[-1], layers[-1][1].in_features)
            self.assertEqual(out_dims, layers[-1][1].out_features)

        # Default values
        input_dims = 1000
        hidden_dims = (100, 100)
        output_dims = 10

        check_layers(input_dims, hidden_dims, output_dims)

        input_dims = 100
        hidden_dims = (100, 200)
        output_dims = 20

        check_layers(input_dims, hidden_dims, output_dims)

        input_dims = 100
        hidden_dims = (200, )
        output_dims = 10

        check_layers(input_dims, hidden_dims, output_dims)

        input_dims = 100
        hidden_dims = (200, 100, 50)
        output_dims = 10

        check_layers(input_dims, hidden_dims, output_dims)

        with self.assertRaises(ValueError):
            CClassifierPyTorchMLP(hidden_dims=(0,))
        with self.assertRaises(ValueError):
            CClassifierPyTorchMLP(hidden_dims=tuple())

    def test_model_params(self):
        """Test for model parameters shape."""
        for name, param in self.clf._model.named_parameters():
            if name.endswith(".weight"):
                # We expect weights to be stored as 2D tensors
                self.assertEqual(2, len(param.shape))
            elif name.endswith(".bias"):
                # We expect biases to be stored as 1D tensors
                self.assertEqual(1, len(param.shape))

    def test_optimizer_params(self):
        """Testing optimizer parameters setting."""
        self.logger.info("Testing parameter `weight_decay`")
        clf = CClassifierPyTorchMLP(weight_decay=1e-2)

        self.assertEqual(1e-2, clf._optimizer.defaults['weight_decay'])

        clf.weight_decay = 1e-4
        self.assertEqual(1e-4, clf._optimizer.defaults['weight_decay'])

    def test_save_load_state(self):
        """Test for load_state using state_dict."""
        lr_default = 1e-2
        lr = 30

        # Initializing a CLF with an unusual parameter value
        self.clf = CClassifierPyTorchMLP(learning_rate=lr)
        self.clf.verbose = 2

        self.assertEqual(lr, self.clf.learning_rate)
        self.assertEqual(lr, self.clf._optimizer.defaults['lr'])

        state = self.clf.state_dict()

        # Initializing the clf again using default parameters
        self.clf = CClassifierPyTorchMLP()
        self.clf.verbose = 2

        self.assertEqual(lr_default, self.clf.learning_rate)
        self.assertEqual(lr_default, self.clf._optimizer.defaults['lr'])

        self.clf.load_state(state)

        self.assertEqual(lr, self.clf.learning_rate)
        self.assertEqual(lr, self.clf._optimizer.defaults['lr'])

    def test_predict(self):
        """Test for predict."""
        self.clf.verbose = 1
        self.clf.fit(self.ds)

        labels, scores = self.clf.predict(
            self.ds[50:100, :].X, return_decision_function=True)

        self.logger.info("Labels:\n{:}".format(labels))
        self.logger.info("Scores:\n{:}".format(scores))

        acc = CMetricAccuracy().performance_score(self.ds[50:100, :].Y, labels)
        self.logger.info("Accuracy: {:}".format(acc))

        self.assertEqual(1.0, acc)  # We should always get the same acc

        self.logger.info("Testing NOT softmax-scaled outputs")

        self.clf.softmax_outputs = False

        labels, scores = self.clf.predict(
            self.ds[50:100, :].X, return_decision_function=True)

        self.logger.info("Labels:\n{:}".format(labels))
        self.logger.info("Scores:\n{:}".format(scores))

        acc = CMetricAccuracy().performance_score(self.ds[50:100, :].Y, labels)
        self.logger.info("Accuracy: {:}".format(acc))

        # Accuracy will not change after scaling the outputs
        self.assertEqual(1.0, acc)  # We should always get the same acc

    def test_out_at_layer(self):
        """Test for extracting output at specific layer."""
        self.clf.verbose = 0
        self.clf.fit(self.ds)

        x_ds = self.ds[0, :]
        x, y = x_ds.X, x_ds.Y

        self.logger.info(
            "Deactivate softmax-scaling to easily compare outputs")
        self.clf.softmax_outputs = False

        layer = None
        self.logger.info("Returning output for layer: {:}".format(layer))
        out_predict = self.clf.predict(x, return_decision_function=True)[1]
        out = self.clf.get_layer_output(x, layer=layer)

        self.logger.info("Output of predict:\n{:}".format(out_predict))
        self.logger.info("Output of get_layer_output:\n{:}".format(out))

        self.assertFalse((out_predict.round(4) != out.round(4)).any())

        layer = 'linear1'
        self.logger.info("Returning output for layer: {:}".format(layer))
        out = self.clf.get_layer_output(x, layer=layer)

        self.logger.info("Output of get_layer_output:\n{:}".format(out))

    def _fun_args(self, x, *args):
        return self.clf.decision_function(x, **args[0])

    def _grad_args(self, x, *args):
        """Wrapper needed as gradient_f_x have **kwargs"""
        return self.clf.gradient_f_x(x, **args[0])

    def test_gradient(self):
        """Test for extracting gradient."""
        self.clf.verbose = 0
        self.clf.fit(self.ds)

        x_ds = self.ds[0, :]
        x, y = x_ds.X, x_ds.Y

        layer = None
        self.logger.info("Testing gradients for layer: {:}".format(layer))

        for c in self.ds.classes:

            self.logger.info("Gradient w.r.t. class {:}".format(c))

            grad = self.clf.gradient_f_x(x, y=c, layer=layer)

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
        self.logger.info("Returning output for layer: {:}".format(layer))
        out = self.clf.get_layer_output(x, layer=layer)
        self.logger.info("Returning gradient for layer: {:}".format(layer))
        grad = self.clf.gradient_f_x(x, w=out, layer=layer)

        self.logger.info("Output of gradient_f_x:\n{:}".format(grad))

        self.assertTrue(grad.is_vector_like)
        self.assertEqual(x.size, grad.size)

    def test_properties(self):
        """Test for properties."""
        self.logger.info("Testing before loading state")

        w = self.clf.w
        self.assertEqual(1, w.ndim)
        # We know the number of params (20*40 + 3*40)
        self.assertEqual(920, w.size)

        b = self.clf.b
        self.assertEqual(1, b.ndim)
        # We know the number of params (40 + 3)
        self.assertEqual(43, b.size)

    def test_deepcopy(self):
        """Test for deepcopy."""
        self.clf.verbose = 0

        self.logger.info("Try deepcopy of not-trained classifier...")
        clf2 = self.clf.deepcopy()

        self.assertEqual(None, self.clf.classes)
        self.assertEqual(None, self.clf.n_features)

        self.assertEqual(None, clf2.classes)
        self.assertEqual(None, clf2.n_features)

        self.assertEqual(self.clf.start_epoch, clf2.start_epoch)

        self.logger.info("Try deepcopy of trained classifier...")
        self.clf.fit(self.ds)

        clf2 = self.clf.deepcopy()

        self.assertFalse((self.clf.classes != clf2.classes).any())
        self.assertEqual(self.clf.n_features, clf2.n_features)

        self.assertEqual(3, clf2.classes.size)
        self.assertEqual(21, clf2.n_features)

        self.assertEqual(self.clf.start_epoch, clf2.start_epoch)

        self.logger.info("Try setting different parameters on the copy...")
        clf2.weight_decay = 300
        self.assertNotEqual(clf2.weight_decay, self.clf.weight_decay)


if __name__ == '__main__':
    CUnitTest.main()
