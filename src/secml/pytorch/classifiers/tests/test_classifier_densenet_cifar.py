from secml.utils import CUnitTest

import random
import torch
import torchvision.transforms as transforms

from secml.data.loader import CDataLoaderCIFAR10
from secml.array import CArray
from secml.pytorch.classifiers import CTorchClassifierDenseNetCifar
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


class TestCTorchClassifierDenseNetCifar(CUnitTest):

    @classmethod
    def setUpClass(cls):

        CUnitTest.setUpClass()

        cls._run_train = False  # Training is a long process for dnn, skip

        cls.tr, cls.ts, transform_tr = cls._load_cifar10()

        cls.clf = CTorchClassifierDenseNetCifar(
            n_epoch=1, batch_size=25, train_transform=transform_tr,
            normalizer=CNormalizerMeanSTD(mean=(0.4914, 0.4822, 0.4465),
                                          std=(0.2023, 0.1994, 0.2010)))
        cls.clf.verbose = 2

    @staticmethod
    def _load_cifar10():

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

        self.clf.train(self.tr, warm_start=False, n_jobs=2)

    def test_classify_cifar10(self):
        """Test classify of the CIFAR10 dataset."""
        state = dl_pytorch_model('densenet-bc-L100-K12')
        self.clf.load_state(state, dataparallel=True)

        labels, scores = self.clf.classify(self.ts[50:100, :].X)

        acc = CMetricAccuracy().performance_score(self.ts[50:100, :].Y, labels)
        self.logger.info("Accuracy: {:}".format(acc))

        self.assertEqual(0.92, acc)  # We should always get the same acc

    def test_gradient(self):
        """Test gradient of the CIFAR10 dataset."""
        state = dl_pytorch_model('densenet-bc-L100-K12')
        self.clf.load_state(state, dataparallel=True)
        
        grad = self.clf.gradient_f_x(self.ts.X[100, :], y=3)

        self.logger.info("Gradient:\n{:}".format(grad))
        self.logger.info("Shape: {:}".format(grad.shape))

    def test_fun(self):
        """Test for discriminant_function() and classify() methods."""
        self.logger.info(
            "Test for discriminant_function() and classify() methods.")

        def _check_df_scores(s, n_samples):
            self.assertEqual(type(s), CArray)
            self.assertTrue(s.isdense)
            self.assertEqual(1, s.ndim)
            self.assertEqual((n_samples,), s.shape)
            self.assertEqual(float, s.dtype)

        def _check_classify_scores(l, s, n_samples, n_classes):
            self.assertEqual(type(l), CArray)
            self.assertEqual(type(s), CArray)
            self.assertTrue(l.isdense)
            self.assertTrue(s.isdense)
            self.assertEqual(1, l.ndim)
            self.assertEqual(2, s.ndim)
            self.assertEqual((n_samples,), l.shape)
            self.assertEqual((n_samples, n_classes), s.shape)
            self.assertEqual(int, l.dtype)
            self.assertEqual(float, s.dtype)
            
        state = dl_pytorch_model('densenet-bc-L100-K12')
        self.clf.load_state(state, dataparallel=True)
        
        x = x_norm = self.ts.X[:5, :]
        p = p_norm = self.ts.X[0, :].ravel()
        
        # Normalizing data if a normalizer is defined
        if self.clf.normalizer is not None:
            x_norm = self.clf.normalizer.normalize(x)
            p_norm = self.clf.normalizer.normalize(p)

        # Testing discriminant_function on multiple points

        df_scores_0 = self.clf.discriminant_function(x, label=0)
        self.logger.info(
            "discriminant_function(x, label=0):\n{:}".format(df_scores_0))
        _check_df_scores(df_scores_0, 5)

        df_scores_1 = self.clf.discriminant_function(x, label=1)
        self.logger.info(
            "discriminant_function(x, label=1):\n{:}".format(df_scores_1))
        _check_df_scores(df_scores_1, 5)

        df_scores_2 = self.clf.discriminant_function(x, label=2)
        self.logger.info(
            "discriminant_function(x, label=2):\n{:}".format(df_scores_2))
        _check_df_scores(df_scores_2, 5)

        # Testing _discriminant_function on multiple points

        ds_priv_scores_0 = self.clf._discriminant_function(x_norm, label=0)
        self.logger.info("_discriminant_function(x_norm, label=0):\n"
                         "{:}".format(ds_priv_scores_0))
        _check_df_scores(ds_priv_scores_0, 5)

        ds_priv_scores_1 = self.clf._discriminant_function(x_norm, label=1)
        self.logger.info("_discriminant_function(x_norm, label=1):\n"
                         "{:}".format(ds_priv_scores_1))
        _check_df_scores(ds_priv_scores_1, 5)

        ds_priv_scores_2 = self.clf._discriminant_function(x_norm, label=2)
        self.logger.info("_discriminant_function(x_norm, label=2):\n"
                         "{:}".format(ds_priv_scores_2))
        _check_df_scores(ds_priv_scores_2, 5)

        # Comparing output of public and private

        self.assertFalse((df_scores_0 != ds_priv_scores_0).any())
        self.assertFalse((df_scores_1 != ds_priv_scores_1).any())
        self.assertFalse((df_scores_2 != ds_priv_scores_2).any())

        # Testing classify on multiple points

        labels, scores = self.clf.classify(x)
        self.logger.info(
            "classify(x):\nlabels: {:}\nscores: {:}".format(labels, scores))
        _check_classify_scores(labels, scores, 5, self.clf.n_classes)

        # Comparing output of discriminant_function and classify

        self.assertFalse((df_scores_0 != scores[:, 0].ravel()).any())
        self.assertFalse((df_scores_1 != scores[:, 1].ravel()).any())
        self.assertFalse((df_scores_2 != scores[:, 2].ravel()).any())

        # Testing discriminant_function on single point

        df_scores_0 = self.clf.discriminant_function(p, label=0)
        self.logger.info(
            "discriminant_function(p, label=0):\n{:}".format(df_scores_0))
        _check_df_scores(df_scores_0, 1)

        df_scores_1 = self.clf.discriminant_function(p, label=1)
        self.logger.info(
            "discriminant_function(p, label=1):\n{:}".format(df_scores_1))
        _check_df_scores(df_scores_1, 1)

        df_scores_2 = self.clf.discriminant_function(p, label=2)
        self.logger.info(
            "discriminant_function(p, label=2):\n{:}".format(df_scores_2))
        _check_df_scores(df_scores_2, 1)

        # Testing _discriminant_function on single point

        df_priv_scores_0 = self.clf._discriminant_function(p_norm, label=0)
        self.logger.info("_discriminant_function(p_norm, label=0):\n{:}"
                         "".format(df_priv_scores_0))
        _check_df_scores(df_priv_scores_0, 1)

        df_priv_scores_1 = self.clf._discriminant_function(p_norm, label=1)
        self.logger.info("_discriminant_function(p_norm, label=1):\n{:}"
                         "".format(df_priv_scores_1))
        _check_df_scores(df_priv_scores_1, 1)

        df_priv_scores_2 = self.clf._discriminant_function(p_norm, label=2)
        self.logger.info("_discriminant_function(p_norm, label=2):\n"
                         "{:}".format(df_priv_scores_2))
        _check_df_scores(df_priv_scores_2, 1)

        # Comparing output of public and private

        self.assertFalse((df_scores_0 != df_priv_scores_0).any())
        self.assertFalse((df_scores_1 != df_priv_scores_1).any())
        self.assertFalse((df_scores_2 != df_priv_scores_2).any())

        self.logger.info("Testing classify on single point")

        labels, scores = self.clf.classify(p)
        self.logger.info(
            "classify(p):\nlabels: {:}\nscores: {:}".format(labels, scores))
        _check_classify_scores(labels, scores, 1, self.clf.n_classes)

        # Comparing output of discriminant_function and classify

        self.assertFalse(
            (df_scores_0 != CArray(scores[:, 0]).ravel()).any())
        self.assertFalse(
            (df_scores_1 != CArray(scores[:, 1]).ravel()).any())
        self.assertFalse(
            (df_scores_2 != CArray(scores[:, 2]).ravel()).any())


if __name__ == '__main__':
    CUnitTest.main()
