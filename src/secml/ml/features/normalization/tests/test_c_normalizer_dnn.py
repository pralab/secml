from secml.ml.features.tests import CPreProcessTestCases

from collections import OrderedDict

try:
    import torch
    import torchvision
except ImportError:
    CPreProcessTestCases.importskip("torch")
    CPreProcessTestCases.importskip("torchvision")
else:
    import torch
    from torch import nn, optim
    from torchvision import transforms
    torch.manual_seed(0)

from secml.array import CArray
from secml.ml.features.normalization import CNormalizerDNN
from secml.ml.classifiers import CClassifierPyTorch
from secml.data.loader import CDLRandom
from secml.optim.function import CFunction


def mlp(input_dims=100, hidden_dims=(50, 50), output_dims=10):
    """Multi-layer Perceptron"""
    if len(hidden_dims) < 1:
        raise ValueError("at least one hidden dim should be defined")
    if any(d <= 0 for d in hidden_dims):
        raise ValueError("each hidden layer must have at least one neuron")

    # Input layers
    layers = [
        ('linear1', torch.nn.Linear(input_dims, hidden_dims[0])),
        ('relu1', torch.nn.ReLU()),
    ]
    # Appending additional hidden layers
    for hl_i, hl_dims in enumerate(hidden_dims[1:]):
        prev_hl_dims = hidden_dims[hl_i]  # Dims of the previous hl
        i_str = str(hl_i + 2)
        layers += [
            ('linear' + i_str, torch.nn.Linear(prev_hl_dims, hl_dims)),
            ('relu' + i_str, torch.nn.ReLU())]
    # Output layers
    layers += [
        ('linear' + str(len(hidden_dims) + 1),
         torch.nn.Linear(hidden_dims[-1], output_dims))]

    # Creating the model with the list of layers
    return torch.nn.Sequential(OrderedDict(layers))


class TestCNormalizerPyTorch(CPreProcessTestCases):

    @classmethod
    def setUpClass(cls):
        cls.ds = CDLRandom(n_samples=40, n_classes=3,
                           n_features=20, n_informative=15,
                           random_state=0).load()

        model = mlp(input_dims=20, hidden_dims=(40,), output_dims=3)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-1)
        cls.net = CClassifierPyTorch(model=model, loss=loss,
                                     optimizer=optimizer, random_state=0,
                                     epochs=10, pretrained=True)
        cls.net.fit(cls.ds)
        cls.norm = CNormalizerDNN(net=cls.net)

        CPreProcessTestCases.setUpClass()

    def test_normalization(self):
        """Testing normalization."""
        x = self.ds.X[0, :]

        self.logger.info("Testing normalization at last layer")

        self.norm.out_layer = None

        out_norm = self.norm.transform(x)
        out_net = self.net.get_layer_output(x, layer=None)

        self.logger.info("Output of normalize:\n{:}".format(out_norm))
        self.logger.info("Output of net:\n{:}".format(out_net))

        self.assert_allclose(out_norm, out_net)

        self.norm.out_layer = 'linear1'

        self.logger.info(
            "Testing normalization at layer {:}".format(self.norm.out_layer))

        out_norm = self.norm.transform(x)
        out_net = self.net.get_layer_output(x, layer=self.norm.out_layer)

        self.logger.info("Output of normalize:\n{:}".format(out_norm))
        self.logger.info("Output of net:\n{:}".format(out_net))

        self.assert_allclose(out_norm, out_net)

    def test_chain(self):
        """Test for preprocessors chain."""
        # Inner preprocessors should be passed to the pytorch clf
        with self.assertRaises(ValueError):
            CNormalizerDNN(net=self.net, preprocess='min-max')

    def test_gradient(self):
        """Test for gradient."""
        x = self.ds.X[0, :]

        layer = None
        self.norm.out_layer = layer
        self.logger.info("Returning gradient for layer: {:}".format(layer))
        shape = self.norm.transform(x).shape
        w = CArray.zeros(shape=shape)
        w[0] = 1
        grad = self.norm.gradient(x, w=w)

        self.logger.info("Output of gradient_f_x:\n{:}".format(grad))

        self.assertTrue(grad.is_vector_like)
        self.assertEqual(x.size, grad.size)

        layer = 'linear1'
        self.norm.out_layer = layer
        self.logger.info("Returning output for layer: {:}".format(layer))
        out = self.net.get_layer_output(x, layer=layer)
        self.logger.info("Returning gradient for layer: {:}".format(layer))
        grad = self.norm.gradient(x, w=out)

        self.logger.info("Output of grad_f_x:\n{:}".format(grad))

        self.assertTrue(grad.is_vector_like)
        self.assertEqual(x.size, grad.size)

    def test_aspreprocess(self):
        """Test for normalizer used as preprocess."""
        from secml.ml.classifiers import CClassifierSVM
        from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA

        model = mlp(input_dims=20, hidden_dims=(40,), output_dims=3)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-1)
        net = CClassifierPyTorch(model=model, loss=loss,
                                 optimizer=optimizer, random_state=0,
                                 epochs=10, preprocess='min-max')
        net.fit(self.ds)

        norm = CNormalizerDNN(net=net)

        clf = CClassifierMulticlassOVA(
            classifier=CClassifierSVM, preprocess=norm)

        self.logger.info("Testing last layer")

        clf.fit(self.ds)

        y_pred, scores = clf.predict(
            self.ds.X, return_decision_function=True)
        self.logger.info("TRUE:\n{:}".format(self.ds.Y.tolist()))
        self.logger.info("Predictions:\n{:}".format(y_pred.tolist()))
        self.logger.info("Scores:\n{:}".format(scores))

        x = self.ds.X[0, :]

        self.logger.info("Testing last layer gradient")

        for c in self.ds.classes:
            self.logger.info("Gradient w.r.t. class {:}".format(c))

            grad = clf.grad_f_x(x, y=c)

            self.logger.info("Output of grad_f_x:\n{:}".format(grad))

            check_grad_val = CFunction(
                clf.decision_function, clf.grad_f_x).check_grad(
                    x, y=c, epsilon=1e-1)
            self.logger.info(
                "norm(grad - num_grad): %s", str(check_grad_val))
            self.assertLess(check_grad_val, 1e-3)

            self.assertTrue(grad.is_vector_like)
            self.assertEqual(x.size, grad.size)

        layer = 'linear1'
        norm.out_layer = layer

        self.logger.info("Testing layer {:}".format(norm.out_layer))

        clf.fit(self.ds)

        y_pred, scores = clf.predict(
            self.ds.X, return_decision_function=True)
        self.logger.info("TRUE:\n{:}".format(self.ds.Y.tolist()))
        self.logger.info("Predictions:\n{:}".format(y_pred.tolist()))
        self.logger.info("Scores:\n{:}".format(scores))

        self.logger.info("Testing 'linear1' layer gradient")
        grad = clf.grad_f_x(x, y=0)  # y is required for multiclassova
        self.logger.info("Output of grad_f_x:\n{:}".format(grad))

        self.assertTrue(grad.is_vector_like)
        self.assertEqual(x.size, grad.size)


if __name__ == '__main__':
    CPreProcessTestCases.main()
