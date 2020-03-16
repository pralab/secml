from secml.ml.classifiers.pytorch.tests import CClassifierPyTorchTestCases

try:
    import torch
    import torchvision
except ImportError:
    CClassifierPyTorchTestCases.importskip("torch")
    CClassifierPyTorchTestCases.importskip("torchvision")
else:
    from torch import nn, optim
    from torchvision import transforms

from secml.data.loader import CDLRandom
from secml.data.splitter import CTrainTestSplit
from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.features import CNormalizerMinMax


class Net(nn.Module):

    def __init__(self, n_features, n_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc2 = nn.Linear(10, n_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestCClassifierPyTorchBlobs(CClassifierPyTorchTestCases):
    """Unittests for CClassifierPyTorch using a FC net and Blobs ds."""

    @classmethod
    def setUpClass(cls):

        CClassifierPyTorchTestCases.setUpClass()

        # Dataset parameters
        cls.n_tr = 200
        cls.n_ts = 100
        cls.n_classes = 3
        cls.n_features = 5

        # Load dataset and split tr/ts
        cls.tr, cls.ts = cls._create_tr_ts(
            cls.n_tr, cls.n_ts, cls.n_classes, cls.n_features)

        # Model and classifier parameters
        cls.batch_size = 20

        # Create the PyTorch model and our classifier
        cls.clf = cls._create_clf(
            cls.n_features, cls.n_classes, cls.batch_size)

        # Train the classifier
        cls.clf.fit(cls.tr)

    @staticmethod
    def _create_tr_ts(n_tr, n_ts, n_classes, n_features):
        """Create BLOBS training and test sets."""
        # generate synthetic data
        ds = CDLRandom(n_samples=n_tr + n_ts,
                       n_classes=n_classes,
                       n_features=n_features,
                       n_redundant=0, n_clusters_per_class=1,
                       class_sep=1, random_state=0).load()

        # Split in training and test
        splitter = CTrainTestSplit(train_size=n_tr,
                                   test_size=n_ts,
                                   random_state=0)
        tr, ts = splitter.split(ds)

        nmz = CNormalizerMinMax()
        tr.X = nmz.fit_transform(tr.X)
        ts.X = nmz.transform(ts.X)

        return tr, ts

    @staticmethod
    def _create_clf(n_features, n_classes, batch_size):
        """Create a FC test network.

        Parameters
        ----------
        n_features : int
        n_classes : int
        batch_size : int

        """
        torch.manual_seed(0)
        net = Net(n_features=n_features, n_classes=n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
        optimizer_scheduler = \
            torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 8], gamma=0.1)

        return CClassifierPyTorch(model=net,
                                  loss=criterion,
                                  optimizer=optimizer,
                                  optimizer_scheduler=optimizer_scheduler,
                                  epochs=10,
                                  batch_size=batch_size,
                                  random_state=0)

    def test_classification(self):
        """Test for `.decision_function` and `.predict` methods."""
        self._test_predict(self.clf, self.ts)
        self._test_accuracy(self.clf, self.ts)

    def test_layer_names(self):
        """Check behavior of `.layer_names` property."""
        self._test_layer_names(self.clf)

    def test_layer_shapes(self):
        """Check behavior of `.layer_shapes` property."""
        self._test_layer_shapes(self.clf)

    def test_get_params(self):
        """Test for `.get_params` method."""
        self.logger.info("params: {}".format(self.clf.get_params()))

    def test_set_params(self):
        """Test for `.set_params` method."""
        self._test_set_params(self.clf, self.tr)

    def test_out_at_layer(self):
        """Test for extracting output at specific layer."""
        self._test_out_at_layer(self.clf, self.ts.X[0, :], layer_name="fc1")

    def test_grad(self):
        """Test for `.gradient` method."""
        self._test_gradient_numerical(
            self.clf, self.ts.X[0, :], th=1e-2, epsilon=1e-3)
        self._test_grad_atlayer(
            self.clf, self.ts.X[0, :], layer_names=["fc1", 'fc2', None])

    def test_softmax_outputs(self):
        """Check behavior of `softmax_outputs` parameter."""
        self._test_softmax_outputs(
            self.clf, self.ts.X[0, :])
        self._test_gradient_numerical(
            self.clf, self.ts.X[0, :], th=1e-2, epsilon=1e-3)

    def test_save_load_model(self):
        """Test for `.save_model` and `.load_model` methods."""
        # Create a second target classifier
        clf_new = self._create_clf(
            self.n_features, self.n_classes, self.batch_size)
        self._test_save_load_model(self.clf, clf_new, self.ts)

    def test_get_set_state(self):
        """Test for `.get_state` and `.set_state` methods."""
        # Create a second target classifier
        clf_new = self._create_clf(
            self.n_features, self.n_classes, self.batch_size)
        self._test_get_set_state(self.clf, clf_new, self.ts)


if __name__ == '__main__':
    CClassifierPyTorchTestCases.main()
