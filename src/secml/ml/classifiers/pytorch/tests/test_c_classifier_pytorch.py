import os
from secml.array import CArray
from secml.ml.classifiers.tests import CClassifierTestCases

try:
    import torch
    import torchvision
except ImportError:
    CClassifierTestCases.importskip("torch")
    CClassifierTestCases.importskip("torchvision")
else:
    from torch import nn, optim
    from torchvision import transforms

from secml.ml.peval.metrics import CMetric


class TestCClassifierPyTorch(CClassifierTestCases):
    """Unittests for CClassifierPyTorch."""

    def setUp(self):
        self.n_classes = 3
        self.n_features = 5
        self.n_samples_tr = 500  # number of training set samples
        self.n_samples_ts = 100  # number of testing set samples
        self.batch_size = 20

    def _test_get_params(self):
        self.logger.info("Testing get params")
        self.logger.debug("params: {}".format(self.clf.get_params()))

    def _test_performance(self):
        """Compute the classifier performance"""
        self.logger.info("Testing PyTorch model accuracy")

        self.assertTrue(self.clf.is_fitted())

        label_torch, y_torch = self.clf.predict(
            self.ts.X, return_decision_function=True)

        acc_torch = CMetric.create('accuracy').performance_score(self.ts.Y,
                                                                 label_torch)

        self.logger.info("Accuracy of PyTorch Model: {:}".format(acc_torch))
        self.assertGreaterEqual(acc_torch, 0.0,
                                "Accuracy of PyTorch Model: {:}".format(acc_torch))

    def _test_predict(self):
        """Confirm that the decision function works."""
        self.logger.info("Testing Predict")

        self.assertTrue(self.clf.is_fitted())

        label_torch, y_torch = self.clf.predict(
            self.ts.X, return_decision_function=True)
        pred = self.clf.decision_function(self.ts.X)

        self.assertTrue(sum(y_torch - pred) < 1e-12)

    def _test_grad_x(self, layer_names):
        """Test for extracting gradient."""
        # TODO add test numerical gradient
        self.logger.info("Testing gradients")

        self.assertTrue(self.clf.is_fitted())

        x_ds = self.ts[0, :]
        x, y = x_ds.X, x_ds.Y

        # Test gradient at specific layers
        for layer in layer_names:
            self.logger.info("Returning gradient for layer: {:}".format(layer))

            # construct w
            shape = self.clf.get_layer_output(x, layer).shape
            w_in = CArray.zeros(shape=(shape))
            w_in[1] = 1

            # call grad
            grad = self.clf.get_layer_gradient(x, w=w_in, layer=layer)

            self.logger.debug("Output of grad_f_x: {:}".format(grad))

            self.assertTrue(grad.is_vector_like)

    def _test_out_at_layer(self, layer_name):
        """Test for extracting output at specific layer."""
        self.logger.info("Testing layer outputs")
        self.assertTrue(self.clf.is_fitted())

        x_ds = self.ts[0, :]
        x, y = x_ds.X, x_ds.Y

        if self.clf.softmax_outputs is True:
            self.logger.info("Deactivate softmax-scaling to easily compare outputs")
            self.clf.softmax_outputs = False

        layer = layer_name
        self.logger.info("Returning output for layer: {:}".format(layer))
        out = self.clf.get_layer_output(x, layer=layer)
        out = out[:10]
        self.logger.debug("Output of get_layer_output: {:}".format(out))

        if layer is None:
            self.assertTrue(
                (self.clf.get_layer_output(x, layer=layer) -
                 self.clf.decision_function(x)).sum() == 0)
            last_layer_name = self.clf.layer_names[-1]
            self.assertTrue(
                (self.clf.get_layer_output(x, layer=last_layer_name) -
                 self.clf.decision_function(x)).sum() == 0)

    def _test_layer_names(self):
        self.logger.info("Testing layers property")
        self.assertTrue(len(list(self.clf.layer_names)) >= 1)
        self.logger.info("Layers: " + ", ".join(self.clf.layer_names))

    def _test_layer_shapes(self):
        self.logger.info("Testing layer shapes property")
        layer_shapes = self.clf.layer_shapes
        for i in layer_shapes:
            self.logger.info("Layer {}: shape {}".format(i, layer_shapes[i]))

    def _test_set_params(self):
        self.logger.info("Testing set params")

        clf_copy = self.clf.deepcopy()

        self.logger.info("Testing assignment on optimizer")
        clf_copy.lr = 10
        self.logger.debug("params: {}".format(clf_copy.get_params()))
        self.assertTrue(clf_copy.get_params()['optimizer']['lr'] == 10)
        self.assertTrue(clf_copy.lr == 10)

        self.logger.info("Testing assignment on model layer")
        clf_copy.fc2 = torch.nn.Linear(
            clf_copy._model.fc2.in_features,
            clf_copy._model.fc2.out_features)

        self.assertTrue(clf_copy.predict(self.tr[0, :].X).size ==
                        self.clf.predict(self.tr[0, :].X).size)

        clf_copy.fit(self.tr)
        self.assertNotEqual(id(self.clf._optimizer), id(clf_copy._optimizer))

        clf_copy.fc2 = torch.nn.Linear(20, 20)
        self.assertTrue(clf_copy._model.fc2.in_features == 20)
        self.assertTrue(clf_copy._model.fc2.out_features == 20)
        self.logger.debug("Copy of the model modified. Last layer should have dims 20x20")
        self.logger.debug("Last layer of copied model: {}".format(clf_copy._model._modules['fc2']))

    def _test_softmax_outputs(self):
        sample = self.tr[0, :].X
        _, scores = self.clf.predict(sample, return_decision_function=True)

        # Store current state of softmax_outputs parameter
        softmax_outputs = self.clf.softmax_outputs

        self.clf.softmax_outputs = True
        _, preds = self.clf.predict(sample, return_decision_function=True)

        self.assertNotEqual(scores.sum(), 1.0)
        self.assert_approx_equal(preds.sum(), 1.0)

        # test gradient
        w_in = CArray.zeros(shape=(self.clf.n_classes, ))
        w_in[1] = 1

        grad = self.clf.gradient(sample, w=w_in)
        self.logger.info("Output of grad_f_x: {:}".format(grad))

        self.assertTrue(grad.is_vector_like)

        # Restore original softmax_outputs parameter
        self.clf.softmax_outputs = softmax_outputs

    def _test_save_load(self, model_creation_fn):
        fname = "state.tar"
        self.clf.save_model(fname)
        self.assertTrue(os.path.exists(fname))
        del self.clf
        model_creation_fn()
        self.clf.load_model(fname)
        self.logger.info("Testing restored model")
        # test that predict works even if no loss and optimizer have been defined
        loss_backup = self.clf._loss
        optimizer_backup = self.clf._optimizer
        optimizer_scheduler_backup = self.clf._optimizer_scheduler
        del self.clf._loss
        del self.clf._optimizer
        del self.clf._optimizer_scheduler
        self._test_performance()
        self.clf._loss = loss_backup
        self.clf._optimizer = optimizer_backup
        self.clf._optimizer_scheduler = optimizer_scheduler_backup

        os.remove(fname)


if __name__ == '__main__':
    TestCClassifierPyTorch.main()
