import os
from collections import OrderedDict

import torchvision

from secml.testing import CUnitTest

try:
    import torch
except ImportError:
    CUnitTest.importskip("torch")
else:
    import torch
    from torch import nn, optim
    from torchvision import transforms

from secml.data.loader import CDLRandom, CDataLoaderMNIST
from secml.data.splitter import CTrainTestSplit
from secml.ml.classifiers.c_classifier_pytorch import CClassifierPyTorch
from secml.ml.features import CNormalizerMinMax
from secml.ml.peval.metrics import CMetric


class Net(nn.Module):
    """
    Model with input size (-1, 5) for blobs dataset
    with 5 features
    """

    def __init__(self, n_features, n_classes):
        """Example network."""
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc2 = nn.Linear(10, n_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

od = OrderedDict([
    ('conv1', nn.Conv2d(1, 10, kernel_size=5)),
    ('pool1', nn.MaxPool2d(2)),
    ('conv2', nn.Conv2d(10, 20, kernel_size=5)),
    ('drop', nn.Dropout2d()),
    ('pool2', nn.MaxPool2d(2)),
    ('flatten', Flatten()),
    ('fc1', nn.Linear(320, 50)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(50, 3)),
])

mnist_net_od = nn.Sequential(OrderedDict(od))

mnist_net = mnist_net_od


print(od.items())


class TestCClassifierPyTorch(CUnitTest):
    """Unittests for CClassifierPyTorch."""

    def setUp(self):
        self.n_classes = 3
        self.n_features = 5
        self.n_samples_tr = 1000  # number of training set samples
        self.n_samples_ts = 500  # number of testing set samples
        self.batch_size = 20

    def _dataset_creation_blobs(self):
        # generate synthetic data
        dataset = CDLRandom(n_samples=self.n_samples_tr + self.n_samples_ts,
                            n_classes=self.n_classes,
                            n_features=self.n_features, n_redundant=0,
                            n_clusters_per_class=1,
                            class_sep=1, random_state=0).load()

        # Split in training and test
        splitter = CTrainTestSplit(train_size=self.n_samples_tr,
                                   test_size=self.n_samples_ts,
                                   random_state=0)
        self.tr, self.ts = splitter.split(dataset)

        # Normalize the data
        nmz = CNormalizerMinMax()
        self.tr.X = nmz.fit_transform(self.tr.X)
        self.ts.X = nmz.transform(self.ts.X)

    def _dataset_creation_mnist(self):
        digits = (1, 5, 9)
        dataset = CDataLoaderMNIST().load('training', digits=digits)

        # Split in training and test
        splitter = CTrainTestSplit(train_size=self.n_samples_tr,
                                   test_size=self.n_samples_ts,
                                   random_state=0)
        self.tr, self.ts = splitter.split(dataset)

        # Normalize the data
        self.tr.X /= 255
        self.ts.X /= 255

    def _dataset_creation_resnet(self):
        dataset = CDLRandom(n_samples=10, n_features=3*224*224).load()

        # Split in training and test
        splitter = CTrainTestSplit(train_size=8,
                                   test_size=2,
                                   random_state=0)
        self.tr, self.ts = splitter.split(dataset)

        # Normalize the data
        nmz = CNormalizerMinMax()
        self.tr.X = nmz.fit_transform(self.tr.X)
        self.ts.X = nmz.transform(self.ts.X)


    def _model_creation_blobs(self):
        net = Net(n_features=self.n_features, n_classes=self.n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),
                              lr=0.1, momentum=0.9)

        self.clf = CClassifierPyTorch(torch_model=net,
                                      loss=criterion,
                                      optimizer=optimizer,
                                      epochs=10,
                                      batch_size=self.batch_size)

    def _model_creation_mnist(self):
        net = mnist_net
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),
                              lr=0.001, momentum=0.9)

        self.clf = CClassifierPyTorch(torch_model=net,
                                      loss=criterion,
                                      optimizer=optimizer,
                                      epochs=10,
                                      batch_size=self.batch_size,
                                      input_shape=(1, 28, 28))

    def _model_creation_resnet(self):
        net = torchvision.models.resnet18(pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),
                              lr=0.001, momentum=0.9)

        self.clf = CClassifierPyTorch(torch_model=net,
                                      loss=criterion,
                                      optimizer=optimizer,
                                      epochs=10,
                                      batch_size=self.batch_size,
                                      input_shape=(3, 224, 224))

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

    def _test_grad_x(self):
        """Test for extracting gradient."""
        # TODO add test numerical gradient
        self.logger.info("Testing gradients")

        self.assertTrue(self.clf.is_fitted())

        x_ds = self.ts[0, :]
        x, y = x_ds.X, x_ds.Y

        # Test gradient at specific layers
        for layer in ['fc1', 'fc2', None]:
            out = self.clf.get_layer_output(x, layer=layer)
            self.logger.info("Returning gradient for layer: {:}".format(layer))
            grad = self.clf.grad_f_x(x, w=out, layer=layer)

            self.logger.debug("Output of grad_f_x: {:}".format(grad))

            self.assertTrue(grad.is_vector_like)
            self.assertEqual(x.size, grad.size)

    def _test_out_at_layer(self):
        """Test for extracting output at specific layer."""
        self.logger.info("Testing layer outputs")
        self.assertTrue(self.clf.is_fitted())

        x_ds = self.ts[0, :]
        x, y = x_ds.X, x_ds.Y

        if self.clf.softmax_outputs is True:
            self.logger.info("Deactivate softmax-scaling to easily compare outputs")
            self.clf.softmax_outputs = False

        layer = None
        self.logger.info("Returning output for layer: {:}".format(layer))
        out_predict = self.clf.predict(x, return_decision_function=True)[1]
        out = self.clf.get_layer_output(x, layer=layer)

        self.logger.debug("Output of predict: {:}".format(out_predict))
        self.logger.debug("Output of get_layer_output: {:}".format(str(out)))

        self.assert_allclose(out_predict, out)

        layer = 'fc1'
        self.logger.info("Returning output for layer: {:}".format(layer))
        out = self.clf.get_layer_output(x, layer=layer)

        self.logger.debug("Output of get_layer_output: {:}".format(out))

    def _test_layer_names(self):
        self.logger.info("Testing layers property")
        self.assertTrue(len(list(self.clf.layers)) >= 1)
        self.logger.info("Layers: " + ", ".join(self.clf.layers))

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
        self.clf.softmax_outputs = True
        _, preds = self.clf.predict(sample, return_decision_function=True)
        self.assertNotEqual(scores.sum(), 1.0)
        self.assert_approx_equal(preds.sum(), 1.0)

    def _test_save_load(self, model_creation_fn):
        fname = "state.tar"
        self.clf.save_model(fname)
        self.assertTrue(os.path.exists(fname))
        self.clf = None
        model_creation_fn()
        self.clf.load_model(fname)
        self.logger.info("Testing restored model")
        # test that predict works even if no loss and optimizer have been defined
        self.clf._loss = None
        self.clf._optimizer = None
        self._test_performance()
        os.remove(fname)

    def test_blobs(self):
        self.logger.info("___________________")
        self.logger.info("Testing Blobs Model")
        self.logger.info("___________________")
        self._dataset_creation_blobs()
        self._model_creation_blobs()
        self._test_layer_names()
        self._test_get_params()
        self.clf.fit(self.tr)
        self._test_set_params()
        self._test_performance()
        self._test_predict()
        self._test_out_at_layer()
        self._test_grad_x()
        self._test_softmax_outputs()
        self._test_save_load(self._model_creation_blobs)

    def test_mnist(self):
        self.logger.info("___________________")
        self.logger.info("Testing MNIST Model")
        self.logger.info("___________________")
        self._dataset_creation_mnist()
        self._model_creation_mnist()
        self._test_layer_names()
        self._test_get_params()
        self.clf.fit(self.tr)
        self._test_performance()
        self._test_predict()
        self._test_out_at_layer()
        self._test_grad_x()
        self._test_softmax_outputs()
        self._test_save_load(self._model_creation_mnist)

    def test_big_net(self):
        self.logger.info("___________________")
        self.logger.info("Testing ResNet11 Model")
        self.logger.info("___________________")
        self._dataset_creation_resnet()
        self._model_creation_resnet()
        self._test_layer_names()
        self._test_get_params()
        self._test_out_at_layer()
        self._test_grad_x()
        self._test_softmax_outputs()
        self._test_save_load(self._model_creation_resnet)

if __name__ == '__main__':
    TestCClassifierPyTorch.main()
