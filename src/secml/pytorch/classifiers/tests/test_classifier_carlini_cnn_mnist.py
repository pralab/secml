import torchvision.transforms as transforms

from secml.utils import CUnitTest
from secml.data.loader import CDataLoaderMNIST
from secml.array import CArray
from secml.ml.peval.metrics import CMetricAccuracy
from secml.pytorch.classifiers import CClassifierPyTorchCarliniCNNMNIST


class TestCClassifierPyTorchCarliniCNNMNIST(CUnitTest):

    def setUp(self):
        self.seed = 0

        self._load_mnist()

        self.clf = CClassifierPyTorchCarliniCNNMNIST(
            train_transform=self.transform_train).fit(self.tr)
        self.clf.verbose = 2

    def _load_mnist(self):
        loader = CDataLoaderMNIST()

        self._digits = CArray.arange(10).tolist()
        self.tr = loader.load('training', digits=self._digits)

        print "classes: ", self.tr.classes
        print "features ", self.tr.num_features

        self.ts = loader.load('testing', digits=self._digits)

        # get dataset img dimension
        # TODO: these should not be included in CDataset!
        # they're lost after conversion tosparse
        self.img_w = self.tr.img_w
        self.img_h = self.tr.img_h

        self.tr.X /= 255.0
        self.ts.X /= 255.0

        idx = CArray.arange(0, self.tr.num_samples)
        val_dts_idx = CArray.randsample(idx, 1000, random_state=self.seed)
        self._val = self.tr[val_dts_idx, :]

        tr_dts_idx = CArray.randsample(idx, 5000, random_state=self.seed)
        self.tr = self.tr[tr_dts_idx, :]

        idx = CArray.arange(0, self.ts.num_samples)
        ts_dts_idx = CArray.randsample(idx, 100, random_state=self.seed)
        self.ts = self.ts[ts_dts_idx, :]

        self.transform_train = transforms.Compose([
            transforms.Lambda(lambda x: x.reshape([28, 28, 1])),
            transforms.ToTensor(),
        ])

    def test_classify(self):
        """Test predict"""

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


# fixme: test deepcopy
# fixme: test incremental training
# fixme: test training after clear


if __name__ == '__main__':
    CUnitTest.main()
