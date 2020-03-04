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

from collections import OrderedDict

from secml.data.loader import CDataLoaderMNIST
from secml.data.splitter import CTrainTestSplit
from secml.ml.classifiers import CClassifierPyTorch


class TestCClassifierPyTorchMNIST(CClassifierPyTorchTestCases):
    """Unittests for CClassifierPyTorch using a CONV net and MNIST ds."""

    @classmethod
    def setUpClass(cls):
        CClassifierPyTorchTestCases.setUpClass()

        # Dataset parameters
        cls.n_tr = 200
        cls.n_ts = 100

        # Load dataset and split tr/ts
        cls.tr, cls.ts = cls._create_tr_ts(cls.n_tr, cls.n_ts)

        # Model and classifier parameters
        cls.batch_size = 20

        # Create the PyTorch model and our classifier
        cls.clf = cls._create_clf(cls.batch_size)

        # Train the classifier
        cls.clf.fit(cls.tr)

    @staticmethod
    def _create_tr_ts(n_tr, n_ts):
        """Create MNIST 3C training and test sets."""
        digits = (1, 5, 9)
        ds = CDataLoaderMNIST().load('training', digits=digits)

        # Split in training and test
        splitter = CTrainTestSplit(train_size=n_tr,
                                   test_size=n_ts,
                                   random_state=0)
        tr, ts = splitter.split(ds)

        tr.X /= 255
        ts.X /= 255

        return tr, ts

    @staticmethod
    def _create_clf(batch_size):
        """Create a CONV test network.

        Parameters
        ----------
        batch_size : int

        """
        torch.manual_seed(0)

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

        net = nn.Sequential(OrderedDict(od))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                   milestones=[1, 5, 8],
                                                   gamma=0.1)

        return CClassifierPyTorch(model=net,
                                  loss=criterion,
                                  optimizer=optimizer,
                                  epochs=10,
                                  batch_size=batch_size,
                                  input_shape=(1, 28, 28),
                                  optimizer_scheduler=scheduler,
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

    def test_out_at_layer(self):
        """Test for extracting output at specific layer."""
        self._test_out_at_layer(self.clf, self.ts.X[0, :], layer_name="fc1")

    def test_grad(self):
        """Test for `.gradient` method."""
        # TODO: ADD TEST OF GRADIENT METHOD
        self._test_grad_atlayer(self.clf, self.ts.X[0, :],
                                layer_names=['conv1', 'fc1', 'fc2', None])

    def test_softmax_outputs(self):
        """Check behavior of `softmax_outputs` parameter."""
        self._test_softmax_outputs(self.clf, self.ts.X[0, :])

    def test_save_load_model(self):
        """Test for `.save_model` and `.load_model` methods."""
        # Create a second target classifier
        clf_new = self._create_clf(self.batch_size)
        self._test_save_load_model(self.clf, clf_new, self.ts)

    def test_get_set_state(self):
        """Test for `.get_state` and `.set_state` methods."""
        # Create a second target classifier
        clf_new = self._create_clf(self.batch_size)
        self._test_get_set_state(self.clf, clf_new, self.ts)


if __name__ == '__main__':
    CClassifierPyTorchTestCases.main()
