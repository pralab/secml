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

import tempfile

from secml.array import CArray
from secml.ml.peval.metrics import CMetricAccuracy
from secml.utils import fm


# TODO: reuse test utilities from CClassifierTestCases
class CClassifierPyTorchTestCases(CClassifierTestCases):
    """Unittests interface for CClassifierPyTorch."""

    def _test_predict(self, clf, ts):
        """Check if `.decision_function` and `.predict` return the same.

        Parameters
        ----------
        clf : CClassifierPyTorch
        ts : CDataset

        """
        self.assertTrue(clf.is_fitted())

        pred = clf.decision_function(ts.X)
        label_torch, y_torch = \
            clf.predict(ts.X, return_decision_function=True)

        self.logger.info("Decision Function:\n{}".format(pred))
        self.logger.info("Classify:\n{}".format(y_torch))

        self.assertTrue(sum(y_torch - pred) < 1e-12)

    def _test_accuracy(self, clf, ts):
        """Check classification accuracy on test set.

        Parameters
        ----------
        clf : CClassifierPyTorch
        ts : CDataset

        """
        self.assertTrue(clf.is_fitted())

        label_torch, y_torch = \
            clf.predict(ts.X, return_decision_function=True)

        acc_torch = CMetricAccuracy().performance_score(ts.Y, label_torch)

        self.logger.info("Accuracy of PyTorch Model: {:}".format(acc_torch))
        self.assertGreater(acc_torch, 0.80)

    def _test_grad_atlayer(self, clf, x, layer_names):
        """Test for extracting gradient at specific layers.

        Parameters
        ----------
        clf : CClassifierPyTorch
        x : CArray
        layer_names : list

        """
        self.assertTrue(clf.is_fitted())

        # Test gradient at specific layers
        for layer in layer_names:
            self.logger.info("Returning gradient for layer: {:}".format(layer))

            # construct w
            shape = clf.get_layer_output(x, layer).shape
            w_in = CArray.zeros(shape=shape)
            w_in[1] = 1

            # call grad
            grad = clf.get_layer_gradient(x, w=w_in, layer=layer)

            self.logger.debug("Output of grad_f_x: {:}".format(grad))

            self.assertTrue(grad.is_vector_like)

    def _test_out_at_layer(self, clf, x, layer_name):
        """Test for extracting output at specific layer.

        Parameters
        ----------
        clf : CClassifierPyTorch
        x : CArray
        layer_name : str or None

        """
        self.assertTrue(clf.is_fitted())

        # Store the value of softmax_outputs parameter
        softmax_output = clf.softmax_outputs

        if softmax_output is True:
            self.logger.info(
                "Deactivate softmax-scaling to easily compare outputs")
            clf.softmax_outputs = False

        layer = layer_name
        self.logger.info("Returning output for layer: {:}".format(layer))
        out = clf.get_layer_output(x, layer=layer)
        out = out[:10]
        self.logger.debug("Output of get_layer_output: {:}".format(out))

        if layer is None:
            self.assertTrue(
                (clf.get_layer_output(x, layer=layer) -
                 clf.decision_function(x)).sum() == 0)
            last_layer_name = clf.layer_names[-1]
            self.assertTrue(
                (clf.get_layer_output(x, layer=last_layer_name) -
                 clf.decision_function(x)).sum() == 0)

        # Restore original value of softmax_outputs parameter
        clf.softmax_outputs = softmax_output

    def _test_layer_names(self, clf):
        """Check behavior of `.layer_names` property.

        Parameters
        ----------
        clf : CClassifierPyTorch

        """
        layer_names = clf.layer_names

        self.logger.info("Layers: " + ", ".join(layer_names))

        self.assertTrue(len(list(layer_names)) >= 1)

    def _test_layer_shapes(self, clf):
        """Check behavior of `.layer_shapes` property.

        Parameters
        ----------
        clf : CClassifierPyTorch

        """
        layer_shapes = clf.layer_shapes
        for i in layer_shapes:
            self.logger.info("Layer {}: shape {}".format(i, layer_shapes[i]))

    def _test_set_params(self, clf, tr):
        """Test for `.set_params` method.

        Parameters
        ----------
        clf : CClassifierPyTorch
        tr : CDataset

        """
        # FIXME: why __deepcopy__ is internally called 2 times when this test
        #  is executed with others and 1 time if executed by itself?
        clf_copy = clf.deepcopy()

        self.logger.info("Testing assignment on optimizer")
        clf_copy.lr = 10
        self.logger.debug("params: {}".format(clf_copy.get_params()))
        self.assertTrue(clf_copy.get_params()['optimizer']['lr'] == 10)
        self.assertTrue(clf_copy.lr == 10)

        self.logger.info("Testing assignment on model layer")
        clf_copy.fc2 = torch.nn.Linear(
            clf_copy._model.fc2.in_features,
            clf_copy._model.fc2.out_features)

        self.assertEqual(clf_copy.predict(tr[0, :].X).size,
                         clf.predict(tr[0, :].X).size)

        clf_copy.fit(tr)
        self.assertNotEqual(id(clf._optimizer), id(clf_copy._optimizer))

        clf_copy.fc2 = torch.nn.Linear(20, 20)
        self.assertTrue(clf_copy._model.fc2.in_features == 20)
        self.assertTrue(clf_copy._model.fc2.out_features == 20)
        self.logger.debug(
            "Copy of the model modified. Last layer should have dims 20x20")
        self.logger.debug("Last layer of copied model: {}".format(
            clf_copy._model._modules['fc2']))

    def _test_softmax_outputs(self, clf, x):
        """Check behavior of `softmax_outputs` parameter.

        Parameters
        ----------
        clf : CClassifierPyTorch
        x : CArray

        """
        self.assertTrue(clf.is_fitted())

        _, scores = clf.predict(x, return_decision_function=True)

        # Store current state of softmax_outputs parameter
        softmax_outputs = clf.softmax_outputs

        clf.softmax_outputs = True
        _, preds = clf.predict(x, return_decision_function=True)

        self.assertNotEqual(scores.sum(), 1.0)
        self.assert_approx_equal(preds.sum(), 1.0)

        # test gradient
        w_in = CArray.zeros(shape=(clf.n_classes, ))
        w_in[1] = 1

        grad = clf.gradient(x, w=w_in)
        self.logger.info("Output of grad_f_x: {:}".format(grad))

        self.assertTrue(grad.is_vector_like)

        # Restore original softmax_outputs parameter
        clf.softmax_outputs = softmax_outputs

    def _test_save_load_model(self, clf, clf_new, ts):
        """Test for `.save_model` and `.load_model` methods.

        Parameters
        ----------
        clf : CClassifierPyTorch
        clf_new : CClassifierPyTorch
            Another instance of the same classifier.
        ts : CDataset

        """
        self.assertTrue(clf.is_fitted())

        pred_y = clf.predict(ts.X)
        self.logger.info(
            "Predictions of the original clf:\n{:}".format(pred_y))

        state_path = fm.join(tempfile.gettempdir(), "state.tar")

        clf.save_model(state_path)

        clf_new.load_model(state_path)

        self.logger.info("Testing restored model")

        # test if predict works even without loss and optimizer
        del clf_new._loss
        del clf_new._optimizer
        del clf_new._optimizer_scheduler

        pred_y_post = clf_new.predict(ts.X)
        self.logger.info(
            "Predictions of the restored model:\n{:}".format(pred_y_post))

        self.assert_array_equal(pred_y, pred_y_post)

        fm.remove_file(state_path)

    def _test_get_set_state(self, clf, clf_new, ts):
        """Test for `.get_state` and `.set_state` methods.

        Parameters
        ----------
        clf : CClassifierPyTorch
        clf_new : CClassifierPyTorch
            Another instance of the same classifier.
        ts : CDataset

        """
        self.assertTrue(clf.is_fitted())

        pred_y = clf.predict(ts.X)
        self.logger.info(
            "Predictions before restoring state:\n{:}".format(pred_y))
        state = clf.get_state(return_optimizer=False)

        # Restore state
        clf_new.set_state(state)

        pred_y_post = clf_new.predict(ts.X)
        self.logger.info(
            "Predictions after restoring state:\n{:}".format(pred_y_post))

        self.assert_array_equal(pred_y, pred_y_post)
