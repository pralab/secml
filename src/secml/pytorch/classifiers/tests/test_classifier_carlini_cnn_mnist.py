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

        self.clf = CClassifierPyTorchCarliniCNNMNIST(random_state=0,
                                                     train_transform=self.transform_train)
        self.clf.verbose = 0  # 2

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
            transforms.Lambda(lambda x: x.reshape([1, 28, 28])),
        ])

    def _get_accuracy(self, clf):
        """
        Compute the classifier accuracy on a subset of test samples

        Returns
        -------
        acc: float
            classifier accuracy
        """
        labels, scores = self.clf.predict(
            self.ts[50:100, :].X, return_decision_function=True)

        self.logger.info("Labels:\n{:}".format(labels))
        self.logger.info("Scores:\n{:}".format(scores))

        acc = CMetricAccuracy().performance_score(self.ts[50:100, :].Y, labels)

        return acc

    def test_accuracy(self):
        """Test the classifier accuracy"""

        self.clf.fit(self.tr)

        acc = self._get_accuracy(self.clf)

        self.logger.info("Accuracy: {:}".format(acc))

        self.assertGreater(acc, 0.90)

        self.logger.info("Testing softmax-scaled outputs")

        self.clf.softmax_outputs = True

        acc2 = self._get_accuracy(self.clf)

        self.logger.info("Accuracy: {:}".format(acc2))

        # Accuracy should not change after scaling the outputs
        self.assertEqual(acc, acc2, "The accuracy is different if we do "
                                    "not scale the logit using softmax")

    def test_deepcopy(self):
        """
        Make a deepcopy of the network, train both and check if their
        accuracy is equal.
        """
        self.logger.info("Check the deepcopy")
        clf2 = self.clf.deepcopy()

        self.clf.fit(self.tr[:100, :])
        acc1 = self._get_accuracy(self.clf)

        clf2.fit(self.tr[:100, :])
        acc2 = self._get_accuracy(clf2)

        self.assertLess(abs(acc1 - acc2) < 1e-3, "The accuracy for the "
                                                 "original classifier and "
                                                 "for its deepcopy is "
                                                 "different")

    def test_incremental_training(self):
        """
        Test if after an incremental training the accuracy increases
        """
        self.logger.info("Check the accuracy after an incremental training")

        self.clf.fit(self.tr[:100, :])

        acc1 = self._get_accuracy(self.clf)

        self.clf.fit(self.tr[:100, :], warm_start=True)

        acc2 = self._get_accuracy(self.clf)

        self.assertLess(abs(acc1 - acc2) < 1e-3,
                        "The accuracy did not increase after "
                        "the incremental training")

    def test_training_from_scratch(self):
        """
        Train a network with a fixed random seed. Clear the network. Train
        it again and check if the accuracy is equal to the one that we get
        before.
        """
        self.logger.info("Check the accuracy when the classifier is trained "
                         "from scratch")

        self.clf.fit(self.tr[:100, :])
        # print self.clf.w.shape

        acc1 = self._get_accuracy(self.clf)

        self.logger.info("The accuracy after the first training is equal to "
                         ": {:}".format(acc1))

        self.clf.clear()

        self.clf.fit(self.tr[:100, :])
        # w2 = self.clf.w.deepcopy()

        acc2 = self._get_accuracy(self.clf)

        self.logger.info("The accuracy after the second training is equal "
                         "to: {:} "
                         "".format(acc1))

        self.assertLess(abs(acc1 - acc2) < 1e-3, "The accuracy is "
                                                 "different after the "
                                                 "first and the second training")


if __name__ == '__main__':
    CUnitTest.main()
