from secml.ml.classifiers.tests import CClassifierTestCases

from secml.utils import fm, pickle_utils
from secml.data.loader import CDataLoaderMNIST
from secml.array import CArray
from secml.ml.peval.metrics import CMetricAccuracy
from secml.pytorch.classifiers import CClassifierPyTorchCNNMNIST


class TestCClassifierPyTorchCNNMNIST(CClassifierTestCases):

    @classmethod
    def setUpClass(cls):
        CClassifierTestCases.setUpClass()

        cls.tr, cls.val, cls.ts = cls._load_mnist()

    def setUp(self):

        self.clf = CClassifierPyTorchCNNMNIST(
            num_classes=3, weight_decay=0, epochs=20, batch_size=10,
            learning_rate=0.1, momentum=0, random_state=0)

    @staticmethod
    def _load_mnist():

        digits = [2, 5, 6]
        digits_str = "".join(['{:}-'.format(i) for i in digits[:-1]])
        digits_str += '{:}'.format(digits[-1])

        # FIXME: REMOVE THIS AFTER CDATALOADERS AUTOMATICALLY STORE DS
        tr_file = fm.join(
            fm.abspath(__file__), 'mnist_tr_{:}.gz'.format(digits_str))
        if not fm.file_exist(tr_file):
            loader = CDataLoaderMNIST()
            tr = loader.load('training', digits=digits)
            pickle_utils.save(tr_file, tr)
        else:
            tr = pickle_utils.load(tr_file)

        ts_file = fm.join(
            fm.abspath(__file__), 'mnist_ts_{:}.gz'.format(digits_str))
        if not fm.file_exist(ts_file):
            loader = CDataLoaderMNIST()
            ts = loader.load('testing', digits=digits)
            pickle_utils.save(ts_file, ts)
        else:
            ts = pickle_utils.load(ts_file)

        tr.X /= 255.0
        ts.X /= 255.0

        idx = CArray.arange(tr.num_samples)
        val_dts_idx = CArray.randsample(idx, 50, random_state=0)
        val = tr[val_dts_idx, :]

        tr_dts_idx = CArray.randsample(idx, 100, random_state=0)
        tr = tr[tr_dts_idx, :]

        idx = CArray.arange(ts.num_samples)
        ts_dts_idx = CArray.randsample(idx, 500, random_state=0)
        ts = ts[ts_dts_idx, :]

        return tr, val, ts

    def test_accuracy(self):
        """Test the classifier accuracy"""

        self.logger.info("Check the classifier accuracy")

        self.clf.verbose = 1
        self.clf.fit(self.tr)

        self.clf.verbose = 0

        labels, scores = self.clf.predict(
            self.ts.X, return_decision_function=True)
        acc = CMetricAccuracy().performance_score(self.ts.Y, labels)

        self.logger.info("Accuracy: {:}".format(acc))

        self.assertGreater(acc, 0.9)

        self.logger.info("Testing softmax-scaled outputs")

        self.clf.softmax_outputs = True

        labels, scores = self.clf.predict(
            self.ts.X, return_decision_function=True)
        acc2 = CMetricAccuracy().performance_score(self.ts.Y, labels)

        self.logger.info("Accuracy: {:}".format(acc2))

        # Accuracy should not change after softmax-scaling the outputs
        self.assertEqual(acc, acc2)


if __name__ == '__main__':
    CClassifierTestCases.main()
