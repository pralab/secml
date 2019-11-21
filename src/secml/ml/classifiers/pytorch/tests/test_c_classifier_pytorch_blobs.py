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

from secml.data.loader import CDLRandom
from secml.data.splitter import CTrainTestSplit
from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.features import CNormalizerMinMax


class TestCClassifierPyTorchBlobs(TestCClassifierPyTorch):
    def setUp(self):
        super(TestCClassifierPyTorchBlobs, self).setUp()
        self._dataset_creation_blobs()
        self._model_creation_blobs()
        self.clf.fit(self.tr)

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

    def _model_creation_blobs(self):

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

        net = Net(n_features=self.n_features, n_classes=self.n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),
                              lr=0.1, momentum=0.9)

        self.clf = CClassifierPyTorch(model=net,
                                      loss=criterion,
                                      optimizer=optimizer,
                                      epochs=10,
                                      batch_size=self.batch_size)

    def test_blobs(self):
        self.logger.info("___________________")
        self.logger.info("Testing Blobs Model")
        self.logger.info("___________________")
        self._test_layer_names()
        self._test_layer_shapes()
        self._test_get_params()
        self._test_set_params()
        self._test_performance()
        self._test_predict()
        self._test_out_at_layer(layer_name="fc1")
        self._test_grad_x(layer_names=["fc1", 'fc2', None])
        self._test_softmax_outputs()
        self._test_save_load(self._model_creation_blobs)