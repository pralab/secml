from collections import OrderedDict

from secml.ml.classifiers.pytorch.tests.test_c_classifier_pytorch import TestCClassifierPyTorch
from secml.testing import CUnitTest

try:
    import torch
    import torchvision
except ImportError:
    CUnitTest.importskip("torch")
    CUnitTest.importskip("torchvision")
else:
    from torch import nn, optim
    from torchvision import transforms

from secml.data.loader import CDataLoaderMNIST
from secml.data.splitter import CTrainTestSplit
from secml.ml.classifiers import CClassifierPyTorch


class TestCClassifierPyTorchMNIST(TestCClassifierPyTorch):
    def setUp(self):
        super(TestCClassifierPyTorchMNIST, self).setUp()
        self._dataset_creation_mnist()
        self._model_creation_mnist()
        self.clf.fit(self.tr)

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

    def _model_creation_mnist(self):
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
        optimizer = optim.SGD(net.parameters(),
                              lr=0.001, momentum=0.9)

        self.clf = CClassifierPyTorch(model=net,
                                      loss=criterion,
                                      optimizer=optimizer,
                                      epochs=10,
                                      batch_size=self.batch_size,
                                      input_shape=(1, 28, 28))

    def test_mnist(self):
        self.logger.info("___________________")
        self.logger.info("Testing MNIST Model")
        self.logger.info("___________________")
        self._test_layer_names()
        self._test_layer_shapes()
        self._test_get_params()
        self._test_performance()
        self._test_predict()
        self._test_out_at_layer(layer_name="fc1")
        self._test_grad_x(layer_names=['conv1', 'fc1', 'fc2', None])
        self._test_softmax_outputs()
        self._test_save_load(self._model_creation_mnist)
