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
from secml.array import CArray
from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.features import CNormalizerMinMax

from secml.settings import SECML_PYTORCH_USE_CUDA


class TestCClassifierPyTorchDNN(CClassifierPyTorchTestCases):
    """Unittests for CClassifierPyTorch using a pretrained RESNET18 net."""

    @classmethod
    def setUpClass(cls):

        CClassifierPyTorchTestCases.setUpClass()

        # Load dataset and split tr/ts
        cls.tr, cls.ts = cls._create_tr_ts()

        # Create the PyTorch model and our classifier
        cls.clf = cls._create_clf()

    @staticmethod
    def _create_tr_ts():
        """Create BLOBS training and test sets."""
        ds = CDLRandom(n_samples=30, n_features=3 * 224 * 224).load()

        # Split in training and test
        splitter = CTrainTestSplit(train_size=10,
                                   test_size=20,
                                   random_state=0)
        tr, ts = splitter.split(ds)

        nmz = CNormalizerMinMax()
        tr.X = nmz.fit_transform(tr.X)
        ts.X = nmz.transform(ts.X)

        return tr, ts

    @staticmethod
    def _create_clf():
        """Load a pretrained RESNET18 network and create the wrapping clf."""
        torch.manual_seed(0)
        net = torchvision.models.resnet18(pretrained=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),
                              lr=0.001, momentum=0.9)

        return CClassifierPyTorch(model=net,
                                  loss=criterion,
                                  optimizer=optimizer,
                                  epochs=10,
                                  batch_size=20,
                                  input_shape=(3, 224, 224),
                                  pretrained=True,
                                  random_state=0)

    def test_accuracy(self):
        """Compare classification accuracy of original and wrapped models."""
        X = self.ts.X
        wrapper_model_scores = self.clf.decision_function(X)

        use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA
        device = "cuda" if use_cuda else "cpu"

        # get torch model scores
        X = torch.from_numpy(X.tondarray()).to(device)
        X = X.view(self.ts.num_samples, 3, 224, 224).float()

        pytorch_net_scores = CArray(self.clf.model(X).detach().cpu())

        self.logger.info(pytorch_net_scores)

        # check if the scores are equal
        self.assert_array_almost_equal(
            wrapper_model_scores, pytorch_net_scores,
            err_msg="The scores of the pytorch network "
                    "and the wrapped one  not equal")

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
        x = self.ts.X[0, :]
        self._test_out_at_layer(self.clf, x, "layer4:1:relu")
        self._test_out_at_layer(self.clf, x, 'bn1')
        self._test_out_at_layer(self.clf, x, 'fc')
        self._test_out_at_layer(self.clf, x, None)

    def test_grad(self):
        """Test for `.gradient` method."""
        # TODO: ADD TEST OF GRADIENT METHOD
        self._test_grad_atlayer(self.clf, self.ts.X[0, :], ['fc', None])

    def test_softmax_outputs(self):
        """Check behavior of `softmax_outputs` parameter."""
        self._test_softmax_outputs(self.clf, self.ts.X[0, :])

    def test_save_load_model(self):
        """Test for `.save_model` and `.load_model` methods."""
        # Create a second target classifier
        clf_new = self._create_clf()
        self._test_save_load_model(self.clf, clf_new, self.ts)

    def test_get_set_state(self):
        """Test for `.get_state` and `.set_state` methods."""
        # Create a second target classifier
        clf_new = self._create_clf()
        self._test_get_set_state(self.clf, clf_new, self.ts)


if __name__ == '__main__':
    CClassifierPyTorchTestCases.main()
