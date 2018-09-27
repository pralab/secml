from secml.utils import CUnitTest

import random
import torch
import torchvision.transforms as transforms

from secml.utils import fm
from secml.core.settings import PYTORCH_MODELS_DIR

from secml.data.loader import CDataLoaderCIFAR10
from secml.pytorch.classifiers import CTorchClassifierDenseNet

from secml.pytorch.normalizers import CNormalizerMeanSTD

from secml.pytorch.models import dl_pytorch_model

from secml.peval.metrics import CMetricAccuracy

use_cuda = torch.cuda.is_available()
print "Using CUDA: ", use_cuda

# Random seed
random.seed(999)
torch.manual_seed(999)
if use_cuda:
    torch.cuda.manual_seed_all(999)


class TestCClassifier(CUnitTest):

    def setUp(self):
        self._run_train = False  # Training is a long process for dnn, skip

    def _load_cifar10(self):

        tr, ts = CDataLoaderCIFAR10().load()

        transform_train = transforms.Compose([
            transforms.Lambda(lambda x: x.reshape([3, 32, 32])),
            transforms.Lambda(lambda x: x.transpose([1, 2, 0])),
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        # For test set we only need to normalize in 0-1
        ts.X /= 255.0

        return tr, ts, transform_train

    def test_train_cifar10(self):
        """Test training the classifier on CIFAR10 dataset."""
        if self._run_train is False:
            # Training is a long process for dnn, skip if not necessary
            return

        tr, ts, transform_tr = self._load_cifar10()

        clf = CTorchClassifierDenseNet(batch_size=25, n_epoch=2,
                                       train_transform=transform_tr)
        clf.verbose = 2

        clf.train(tr, warm_start=False, n_jobs=2)

    def test_classify_cifar10(self):
        """Test classify of the CIFAR10 dataset."""
        tr, ts, transform_tr = self._load_cifar10()

        clf = CTorchClassifierDenseNet(batch_size=25, n_epoch=1,
                                       train_transform=transform_tr,
                                       normalizer=CNormalizerMeanSTD(
                                           mean=(0.4914, 0.4822, 0.4465),
                                           std=(0.2023, 0.1994, 0.2010)))
        clf.verbose = 2

        state = dl_pytorch_model('densenet-bc-L100-K12')

        clf.load_state(state, dataparallel=True)

        labels, scores = clf.classify(ts[50:100, :].X)

        acc = CMetricAccuracy().performance_score(ts[50:100, :].Y, labels)
        self.logger.info("Accuracy: {:}".format(acc))

        self.assertEqual(0.92, acc)  # We should always get the same acc

    def test_gradient(self):
        """Test gradient of the CIFAR10 dataset."""
        tr, ts, transform_tr = self._load_cifar10()

        clf = CTorchClassifierDenseNet(batch_size=25, n_epoch=1,
                                       train_transform=transform_tr)
        clf.verbose = 2

        state = dl_pytorch_model('densenet-bc-L100-K12')

        clf.load_state(state, dataparallel=True)

        grad = clf.gradient('x', ts.X[100, :], y=3)

        self.logger.info("Gradient:\n{:}".format(grad))
        self.logger.info("Shape: {:}".format(grad.shape))


if __name__ == '__main__':
    CUnitTest.main()
