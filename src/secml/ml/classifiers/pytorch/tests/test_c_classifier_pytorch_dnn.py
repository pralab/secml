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

class TestCClassifierPyTorchDNN(TestCClassifierPyTorch):
    def setUp(self):
        super(TestCClassifierPyTorchDNN, self).setUp()
        self._dataset_creation_resnet()
        self._model_creation_resnet()

    def _dataset_creation_resnet(self):
        dataset = CDLRandom(n_samples=10, n_features=3 * 224 * 224).load()

        # Split in training and test
        splitter = CTrainTestSplit(train_size=8,
                                   test_size=2,
                                   random_state=0)
        self.tr, self.ts = splitter.split(dataset)

        # Normalize the data
        nmz = CNormalizerMinMax()
        self.tr.X = nmz.fit_transform(self.tr.X)
        self.ts.X = nmz.transform(self.ts.X)

    def _model_creation_resnet(self):
        net = torchvision.models.resnet18(pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),
                              lr=0.001, momentum=0.9)

        self.clf = CClassifierPyTorch(model=net,
                                      loss=criterion,
                                      optimizer=optimizer,
                                      epochs=10,
                                      batch_size=self.batch_size,
                                      input_shape=(3, 224, 224))

    def test_big_net(self):
        self.logger.info("___________________")
        self.logger.info("Testing ResNet11 Model")
        self.logger.info("___________________")
        self._test_layer_names()
        self._test_layer_shapes()
        self._test_get_params()
        self._test_out_at_layer("layer4:1:relu")
        self._test_out_at_layer(['bn1', 'fc'])
        self._test_out_at_layer(None)
        self._test_grad_x(['fc', None])
        self._test_softmax_outputs()
        self._test_save_load(self._model_creation_resnet)