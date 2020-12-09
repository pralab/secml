from secml.ml.classifiers.reject.tests import CClassifierRejectTestCases

try:
    import torch
except ImportError:
    CClassifierRejectTestCases.importskip("torch")
else:
    from torch import nn, optim

from collections import OrderedDict

from secml.ml import CClassifierPyTorch
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.peval.metrics import CMetric
from secml.data.loader import CDataLoaderMNIST
from secml.data.splitter import CTrainTestSplit
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF
from secml.core.constants import inf


class TestCClassifierDNR(CClassifierRejectTestCases):
    """Unit test for CClassifierRejectThreshold."""

    @classmethod
    def setUpClass(cls):
        """Load the dataset and train classifiers"""

        CClassifierRejectTestCases.setUpClass()

        # Load dataset
        tr_dnn, cls.tr, cls.ts = cls._get_dataset()

        # Train the DNN
        cls.dnn = cls._get_dnn(tr_dnn)

        # Create the classifier
        cls.clf = cls._create_clf(cls.dnn)
        cls.clf.fit(cls.tr.X, cls.tr.Y)

    @staticmethod
    def _get_dataset():
        """Load training set used for DNN and DNR, and a test set,
        subsabmpled from MNIST training set"""

        # Load only 4 digits
        digits = (1, 4, 5, 9)
        ds = CDataLoaderMNIST().load('training', digits=digits)

        # Extract training set for DNN and for DNR classifier and test set
        tr_dnn, ds_dnr = CTrainTestSplit(
            train_size=300, test_size=350, random_state=0).split(ds)
        tr_dnr, ts_dnr = CTrainTestSplit(
            train_size=300, test_size=50, random_state=0).split(ds_dnr)

        # Normalize data in [0, 1]
        tr_dnn.X /= 255.
        tr_dnr.X /= 255.
        ts_dnr.X /= 255.

        return tr_dnn, tr_dnr, ts_dnr

    @staticmethod
    def _get_dnn(tr_dnn):
        """Train a simple DNN on MNIST"""

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
            ('fc2', nn.Linear(50, 4)),
        ])

        net = nn.Sequential(OrderedDict(od))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=[1, 5, 8], gamma=0.1)

        dnn = CClassifierPyTorch(
            model=net, loss=criterion, optimizer=optimizer, epochs=10,
            batch_size=20, input_shape=(1, 28, 28),
            optimizer_scheduler=scheduler, random_state=0)
        dnn.fit(tr_dnn.X, tr_dnn.Y)

        return dnn

    @staticmethod
    def _create_clf(dnn):
        """Initialize the DNR classifier passing a single `layer_clf`"""
        layers = ['conv2', 'relu']
        combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=1)
        layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=1)

        return CClassifierDNR(combiner, layer_clf, dnn, layers, -inf)

    @staticmethod
    def _create_clf_dict(dnn):
        """Initialize the DNR classifier passing a `layer_clf` dict"""
        layers = ['conv2', 'relu']
        combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=1)
        layer_clf = {'conv2': CClassifierSVM(kernel=CKernelRBF(gamma=1), C=1),
                     'relu': CClassifierSVM(kernel=CKernelRBF(gamma=1), C=1)}

        return CClassifierDNR(combiner, layer_clf, dnn, layers, -inf)

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        self.logger.info(
            "Test for decision_function() and predict() methods.")

        scores_d = self._test_fun(self.clf, self.ts.todense())
        scores_s = self._test_fun(self.clf, self.ts.tosparse())

        self.assert_array_almost_equal(scores_d, scores_s)

        y_pred = self.clf.predict(self.ts.X)
        accuracy = (self.ts.Y == y_pred).mean()
        self.logger.info("Accuracy: {:}".format(accuracy))

    def test_reject(self):
        y_pred, score_pred = self.clf.predict(
            self.ts.X, return_decision_function=True)
        # set the threshold to have 10% of rejection rate
        threshold = self.clf.compute_threshold(0.1, self.ts)
        self.clf.threshold = threshold
        y_pred_reject, score_pred_reject = self.clf.predict(
            self.ts.X, return_decision_function=True)

        # Compute the number of rejected samples
        n_rej = (y_pred_reject == -1).sum()
        self.logger.info("Rejected samples: {:}".format(n_rej))

        self.logger.info("Real: \n{:}".format(self.ts.Y))
        self.logger.info("Predicted: \n{:}".format(y_pred))
        self.logger.info(
            "Predicted with reject: \n{:}".format(y_pred_reject))

        acc = CMetric.create('accuracy').performance_score(
            y_pred, self.ts.Y)
        self.logger.info("Accuracy no rejection: {:}".format(acc))

        rej_acc = CMetric.create('accuracy').performance_score(
            y_pred_reject[y_pred_reject != -1],
            self.ts.Y[y_pred_reject != -1])
        self.logger.info("Accuracy with rejection: {:}".format(rej_acc))

        # check that the accuracy using reject is higher that the one
        # without rejects
        self.assertGreaterEqual(
            rej_acc, acc, "The accuracy of the classifier that is allowed "
                          "to reject is lower than the one of the "
                          "classifier that is not allowed to reject")

    def test_set_params(self):
        """Test layer classifiers parameters setting"""
        self.clf.set_params({'conv2.C': 10, 'conv2.kernel.gamma': 20})
        self.clf.set('relu.C', 20)

        self.assertEqual(self.clf._layer_clfs['conv2'].C, 10.0)
        self.assertEqual(self.clf._layer_clfs['conv2'].kernel.gamma, 20.0)
        self.assertEqual(self.clf._layer_clfs['relu'].C, 20.0)

    def test_create_dict(self):
        self.logger.info("Testing creation with `layer_clf` dict")
        clf_dict = self._create_clf_dict(self.dnn)
        clf_dict.fit(self.tr.X, self.tr.Y)


if __name__ == '__main__':
    CClassifierRejectTestCases.main()
