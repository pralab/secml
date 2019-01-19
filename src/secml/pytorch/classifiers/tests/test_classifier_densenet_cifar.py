from secml.utils import CUnitTest

import torchvision.transforms as transforms

from secml.data.loader import CDataLoaderCIFAR10
from secml.array import CArray
from secml.pytorch.classifiers import CClassifierPyTorchDenseNetCifar
from secml.pytorch.normalizers import CNormalizerMeanSTD
from secml.pytorch.models import dl_pytorch_model
from secml.ml.peval.metrics import CMetricAccuracy


class TestCClassifierPyTorchDenseNetCifar(CUnitTest):

    @classmethod
    def setUpClass(cls):

        CUnitTest.setUpClass()

        cls._run_train = False  # Training is a long process for dnn, skip

        cls.tr, cls.ts, transform_tr = cls._load_cifar10()

        cls.clf = CClassifierPyTorchDenseNetCifar(
            batch_size=25, epochs=1, train_transform=transform_tr,
            preprocess=CNormalizerMeanSTD(mean=(0.4914, 0.4822, 0.4465),
                                          std=(0.2023, 0.1994, 0.2010)),
            random_state=0)
        cls.clf.verbose = 2

    def setUp(self):

        # Restore as this could have be modified by the test cases
        self.clf.softmax_outputs = False

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

        self.clf.fit(self.tr, warm_start=False, n_jobs=2)

    def test_classify_cifar10(self):
        """Test predict of the CIFAR10 dataset."""
        state = dl_pytorch_model('densenet-bc-L100-K12')
        self.clf.load_state(state, dataparallel=True)

        labels, scores = self.clf.predict(
            self.ts[50:100, :].X, return_decision_function=True)

        self.logger.info("Labels:\n{:}".format(labels))
        self.logger.info("Scores:\n{:}".format(scores))

        acc = CMetricAccuracy().performance_score(self.ts[50:100, :].Y, labels)
        self.logger.info("Accuracy: {:}".format(acc))

        self.assertEqual(0.92, acc)  # We should always get the same acc

        self.logger.info("Testing softmax-scaled outputs")

        self.clf.softmax_outputs = True

        labels, scores = self.clf.predict(
            self.ts[50:100, :].X, return_decision_function=True)

        self.logger.info("Labels:\n{:}".format(labels))
        self.logger.info("Scores:\n{:}".format(scores))

        acc = CMetricAccuracy().performance_score(self.ts[50:100, :].Y, labels)
        self.logger.info("Accuracy: {:}".format(acc))

        # Accuracy will not change after scaling the outputs
        self.assertEqual(0.92, acc)  # We should always get the same acc

    def test_gradient(self):
        """Test gradient of the CIFAR10 dataset."""
        state = dl_pytorch_model('densenet-bc-L100-K12')
        self.clf.load_state(state, dataparallel=True)
        
        grad = self.clf.gradient_f_x(self.ts.X[100, :], y=3)

        self.logger.info("Gradient:\n{:}".format(grad))
        self.logger.info("Shape: {:}".format(grad.shape))

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        self.logger.info(
            "Test for decision_function() and predict() methods.")

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
        
        # Preprocessing data if a preprocess is defined
        if self.clf.preprocess is not None:
            x_norm = self.clf.preprocess.normalize(x)
            p_norm = self.clf.preprocess.normalize(p)

        # Testing decision_function on multiple points

        df_scores_0 = self.clf.decision_function(x, y=0)
        self.logger.info(
            "decision_function(x, y=0):\n{:}".format(df_scores_0))
        _check_df_scores(df_scores_0, 5)

        df_scores_1 = self.clf.decision_function(x, y=1)
        self.logger.info(
            "decision_function(x, y=1):\n{:}".format(df_scores_1))
        _check_df_scores(df_scores_1, 5)

        df_scores_2 = self.clf.decision_function(x, y=2)
        self.logger.info(
            "decision_function(x, y=2):\n{:}".format(df_scores_2))
        _check_df_scores(df_scores_2, 5)

        # Testing _decision_function on multiple points

        ds_priv_scores_0 = self.clf._decision_function(x_norm, y=0)
        self.logger.info("_decision_function(x_norm, y=0):\n"
                         "{:}".format(ds_priv_scores_0))
        _check_df_scores(ds_priv_scores_0, 5)

        ds_priv_scores_1 = self.clf._decision_function(x_norm, y=1)
        self.logger.info("_decision_function(x_norm, y=1):\n"
                         "{:}".format(ds_priv_scores_1))
        _check_df_scores(ds_priv_scores_1, 5)

        ds_priv_scores_2 = self.clf._decision_function(x_norm, y=2)
        self.logger.info("_decision_function(x_norm, y=2):\n"
                         "{:}".format(ds_priv_scores_2))
        _check_df_scores(ds_priv_scores_2, 5)

        # Comparing output of public and private

        self.assertFalse((df_scores_0 != ds_priv_scores_0).any())
        self.assertFalse((df_scores_1 != ds_priv_scores_1).any())
        self.assertFalse((df_scores_2 != ds_priv_scores_2).any())

        # Testing predict on multiple points

        labels, scores = self.clf.predict(x, return_decision_function=True)
        self.logger.info(
            "predict(x):\nlabels: {:}\nscores: {:}".format(labels, scores))
        _check_classify_scores(labels, scores, 5, self.clf.n_classes)

        # Comparing output of decision_function and predict

        self.assertFalse((df_scores_0 != scores[:, 0].ravel()).any())
        self.assertFalse((df_scores_1 != scores[:, 1].ravel()).any())
        self.assertFalse((df_scores_2 != scores[:, 2].ravel()).any())

        # Testing decision_function on single point

        df_scores_0 = self.clf.decision_function(p, y=0)
        self.logger.info(
            "decision_function(p, y=0):\n{:}".format(df_scores_0))
        _check_df_scores(df_scores_0, 1)

        df_scores_1 = self.clf.decision_function(p, y=1)
        self.logger.info(
            "decision_function(p, y=1):\n{:}".format(df_scores_1))
        _check_df_scores(df_scores_1, 1)

        df_scores_2 = self.clf.decision_function(p, y=2)
        self.logger.info(
            "decision_function(p, y=2):\n{:}".format(df_scores_2))
        _check_df_scores(df_scores_2, 1)

        # Testing _decision_function on single point

        df_priv_scores_0 = self.clf._decision_function(p_norm, y=0)
        self.logger.info("_decision_function(p_norm, y=0):\n{:}"
                         "".format(df_priv_scores_0))
        _check_df_scores(df_priv_scores_0, 1)

        df_priv_scores_1 = self.clf._decision_function(p_norm, y=1)
        self.logger.info("_decision_function(p_norm, y=1):\n{:}"
                         "".format(df_priv_scores_1))
        _check_df_scores(df_priv_scores_1, 1)

        df_priv_scores_2 = self.clf._decision_function(p_norm, y=2)
        self.logger.info("_decision_function(p_norm, y=2):\n"
                         "{:}".format(df_priv_scores_2))
        _check_df_scores(df_priv_scores_2, 1)

        # Comparing output of public and private

        self.assertFalse((df_scores_0 != df_priv_scores_0).any())
        self.assertFalse((df_scores_1 != df_priv_scores_1).any())
        self.assertFalse((df_scores_2 != df_priv_scores_2).any())

        self.logger.info("Testing predict on single point")

        labels, scores = self.clf.predict(p, return_decision_function=True)
        self.logger.info(
            "predict(p):\nlabels: {:}\nscores: {:}".format(labels, scores))
        _check_classify_scores(labels, scores, 1, self.clf.n_classes)

        # Comparing output of decision_function and predict

        self.assertFalse(
            (df_scores_0 != CArray(scores[:, 0]).ravel()).any())
        self.assertFalse(
            (df_scores_1 != CArray(scores[:, 1]).ravel()).any())
        self.assertFalse(
            (df_scores_2 != CArray(scores[:, 2]).ravel()).any())

    def test_params(self):
        """Testing parameters setting."""
        self.logger.info("Testing parameter `weight_decay`")
        clf = CClassifierPyTorchDenseNetCifar(weight_decay=1e-2)

        self.assertEqual(1e-2, clf._optimizer.defaults['weight_decay'])

        clf.weight_decay = 1e-4
        self.assertEqual(1e-4, clf._optimizer.defaults['weight_decay'])

    def test_save_load_state(self):
        """Test for load_state using state_dict."""
        lr_default = 1e-2
        lr = 30

        # Initializing a CLF with an unusual parameter value
        self.clf = CClassifierPyTorchDenseNetCifar(learning_rate=lr)
        self.clf.verbose = 2

        self.assertEqual(lr, self.clf.learning_rate)
        self.assertEqual(lr, self.clf._optimizer.defaults['lr'])

        state = self.clf.state_dict()

        # Initializing the clf again using default parameters
        self.clf = CClassifierPyTorchDenseNetCifar()
        self.clf.verbose = 2

        self.assertEqual(lr_default, self.clf.learning_rate)
        self.assertEqual(lr_default, self.clf._optimizer.defaults['lr'])

        self.clf.load_state(state)

        self.assertEqual(lr, self.clf.learning_rate)
        self.assertEqual(lr, self.clf._optimizer.defaults['lr'])


if __name__ == '__main__':
    CUnitTest.main()
