import os

import torchvision.transforms as transforms
import numpy as np

from secml.data import CDataset
from secml.utils import CUnitTest
from secml.data.loader import CDataLoaderMNIST
from secml.array import CArray
from secml.ml.peval.metrics import CMetricAccuracy
from secml.pytorch.classifiers import CClassifierPyTorchCNNMNIST


class TestCClassifierPyTorchCarliniCNNMNIST(CUnitTest):

    def setUp(self):
        self.seed = 0

        self._load_mnist()

        self.clf = CClassifierPyTorchCNNMNIST(random_state=0, num_classes=2,
                                              train_transform=self.transform_train)

        self.clf.verbose = 0  # 2

    def _load_mnist(self):
        loader = CDataLoaderMNIST()

        self._digits = [8, 9]
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
        ts_dts_idx = CArray.randsample(idx, 1000, random_state=self.seed)
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
        labels, scores = clf.predict(
            self.ts.X, return_decision_function=True)

        acc = CMetricAccuracy().performance_score(self.ts.Y, labels)

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

        # check if the parameter are equal comparing their dictionary
        dict_clf_1 = self.clf.get_params()
        dict_clf_2 = clf2.get_params()
        self.assertEqual(dict_clf_1, dict_clf_2, "the dictionary of the two "
                                                 "classifiers is different")

        # check some parameters defined in the classifier
        self.assertEqual(self.clf1.n_features, clf2.n_features,
                         "the number of features parameter is different for "
                         "the two classifiers")

        # check some parameters that are specific of the pytorch network
        self.assertEqual(self.clf1.start_epoch, clf2.start_epoch,
                         "the number of features parameter is different for "
                         "the two classifiers")

        # check the accuracy
        self.clf.fit(self.tr[:100, :])
        acc1 = self._get_accuracy(self.clf)

        clf2.fit(self.tr[:100, :])
        acc2 = self._get_accuracy(clf2)

        self.assertLess(abs(acc1 - acc2) < 1e-3, "The accuracy for the "
                                                 "original classifier and "
                                                 "for its deepcopy is "
                                                 "different")

        # check the classifier weights
        self.assertFalse((self.clf.w.todense() != clf2.w.todense()).any(),
                         "Loaded arrays are not equal!")

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

    def _load_problematic_points(self):

        probl_points_path = "{:}/{:}".format(os.path.dirname(
            os.path.abspath(__file__)), "p_points")
        if os.path.isfile(probl_points_path + '.npz'):
            with np.load(probl_points_path + '.npz') as fm:
                X = fm['X']
                Y = fm['Y']
            X = CArray(X)
            Y = CArray(Y)
            pois_data = CDataset(X, Y)
        else:
            raise ValueError("file not found!")

        return pois_data

    def test_training_from_scratch(self):
        """
        Train a network with a fixed random seed. Clear the network. Train
        it again and check if the accuracy is equal to the one that we get
        before.
        """
        self.logger.info("Check the accuracy when the classifier is trained "
                         "from scratch")

        pp = self._load_problematic_points()
        dts2 = self.tr[:500, :].append(pp)
        print "dts shape ", dts2.X.shape

        self.clf.verbose = 1

        # train the classifier on dataset 2
        self.clf.fit(dts2)

        print "best acc ", self.clf.best_acc
        print "epochs ", self.clf.epochs

        acc_clf1_tr2 = self._get_accuracy(self.clf)

        self.logger.info("acc1: {:}".format(acc_clf1_tr2))

        # train again the classifier on dataset 2
        self.clf.fit(dts2)

        print "best acc ", self.clf.best_acc
        print "epochs ", self.clf.epochs

        acc_clf_tr2b = self._get_accuracy(self.clf)

        self.logger.info("acc2: {:}".format(acc_clf_tr2b))

        self.assertLess(abs(acc_clf1_tr2 - acc_clf_tr2b) , 1e-3,
                        "The accuracy is different after the first and the "
                        "second training on the same dataset")


if __name__ == '__main__':
    CUnitTest.main()
