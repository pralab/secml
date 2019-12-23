from secml.ml.classifiers.pytorch.tests.test_c_classifier_pytorch import \
    TestCClassifierPyTorch
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

from secml.settings import SECML_PYTORCH_USE_CUDA


class TestCClassifierPyTorchDNN(TestCClassifierPyTorch):
    def setUp(self):
        self.logger.info("Testing ResNet11 Model")
        super(TestCClassifierPyTorchDNN, self).setUp()
        self._dataset_creation_resnet()
        self._model_creation_resnet()

    def _dataset_creation_resnet(self):
        dataset = CDLRandom(n_samples=30, n_features=3 * 224 * 224).load()

        # Split in training and test
        splitter = CTrainTestSplit(train_size=8,
                                   test_size=22,
                                   random_state=0)
        self.tr, self.ts = splitter.split(dataset)

        # Normalize the data
        nmz = CNormalizerMinMax()
        self.tr.X = nmz.fit_transform(self.tr.X)
        self.ts.X = nmz.transform(self.ts.X)

    def _model_creation_resnet(self):
        torch.manual_seed(0)
        self.net = torchvision.models.resnet18(pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(),
                              lr=0.001, momentum=0.9)

        self.clf = CClassifierPyTorch(model=self.net,
                                      loss=criterion,
                                      optimizer=optimizer,
                                      epochs=10,
                                      batch_size=self.batch_size,
                                      input_shape=(3, 224, 224),
                                      pretrained=True)

    def test_layer_names(self):
        self._test_layer_names()

    def _test_layer_shapes(self):
        self._test_layer_shapes()

    def test_get_params(self):
        self._test_get_params()

    def test_out_at_layer(self):
        self._test_out_at_layer("layer4:1:relu")
        self._test_out_at_layer('bn1')
        self._test_out_at_layer('fc')
        self._test_out_at_layer(None)

    def test_grad_x(self):
        self._test_grad_x(['fc', None])

    def test_softmax_outputs(self):
        self._test_softmax_outputs()

    def test_save_load(self):
        self._test_save_load(self._model_creation_resnet)

    def test_accuracy(self):
        """
        Check if the accuracy of the torch network and the one of the
        wrapper is the same.
        """
        X = self.ts.X
        wrapper_model_scores = self.clf.decision_function(X)

        use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA
        device = "cuda" if use_cuda else "cpu"

        # get torch model scores
        X = torch.from_numpy(X.tondarray()).to(device)
        X = X.view(self.ts.num_samples, 3, 224, 224).float()
        from secml.array import CArray
        pytorch_net_scores = CArray(self.net(X).detach().cpu())

        self.logger.info(pytorch_net_scores)

        # check if the scores are equal
        self.assert_array_almost_equal(wrapper_model_scores, pytorch_net_scores,
                                err_msg="The scores of the pytorch network "
                                        "and the wrapped one are not equal")
