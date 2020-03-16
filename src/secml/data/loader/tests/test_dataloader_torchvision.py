import logging
from collections import OrderedDict

from secml.testing import CUnitTest

try:
    import torch
    import torchvision
except ImportError:
    CUnitTest.importskip("torch")
    CUnitTest.importskip("torchvision")
else:
    from torch import nn, optim
    from torchvision import datasets

from secml.data.loader import CDataLoaderTorchDataset
from secml.data.splitter import CTrainTestSplit
from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.features import CNormalizerMeanStd
from secml.ml.peval.metrics import CMetric


class TestCDataLoaderTorchDataset(CUnitTest):
    """Unittest for CDataLoaderTorchDataset"""

    def setUp(self):
        self.n_samples_tr = 100  # number of training set samples
        self.n_samples_ts = 50  # number of testing set samples

    def _create_ds(self):
        torchvision_dataset = datasets.MNIST
        ds = CDataLoaderTorchDataset(torchvision_dataset, train=True, download=True).load()
        splitter = CTrainTestSplit(train_size=self.n_samples_tr,
                                   test_size=self.n_samples_ts,
                                   random_state=0)
        self.tr, self.ts = splitter.split(ds)

    def _create_net(self):
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
            ('fc2', nn.Linear(50, 10)),
        ])

        net = nn.Sequential(OrderedDict(od))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),
                              lr=0.001, momentum=0.9)
        preprocess = CNormalizerMeanStd(mean=0.5, std=0.5)

        self.clf = CClassifierPyTorch(model=net,
                                      loss=criterion,
                                      optimizer=optimizer,
                                      epochs=1,
                                      batch_size=100,
                                      input_shape=(1, 28, 28),
                                      preprocess=preprocess)

    def test_train_net(self):
        self._create_ds()
        self._create_net()
        self.clf.fit(self.tr)
        label_torch, y_torch = self.clf.predict(
            self.ts.X, return_decision_function=True)
        acc_torch = CMetric.create('accuracy').performance_score(self.ts.Y,
                                                                 label_torch)
        logging.info("Accuracy: {:.3f}".format(acc_torch))
        self.assertGreater(acc_torch, 0)
