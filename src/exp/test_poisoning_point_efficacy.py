import os

import numpy as np
import torchvision.transforms as transforms

from secml.pytorch.classifiers import CClassifierPyTorchCarliniCNNMNIST
from secml.core import CCreator
from secml.array import CArray
from secml.data import CDataset
from secml.ml.peval.metrics import CMetricAccuracy
from dataset_creation import create_mnist_dataset


class TestPoisoningPointEfficacy(CCreator):
    """
    Test that given a saved set of poisoning points compute their
    effectiveness into decrease the classifier accuracy.
    """

    # def _load_mnist(self):
    #     loader = CDataLoaderMNIST()
    #
    #     self._digits = [8, 9]
    #     self.tr = loader.load('training', digits=self._digits)
    #
    #     print "classes: ", self.tr.classes
    #     print "features ", self.tr.num_features
    #
    #     self.ts = loader.load('testing', digits=self._digits)
    #
    #     # get dataset img dimension
    #     # TODO: these should not be included in CDataset!
    #     # they're lost after conversion tosparse
    #     self.img_w = self.tr.img_w
    #     self.img_h = self.tr.img_h
    #
    #     self.tr.X /= 255.0
    #     self.ts.X /= 255.0
    #
    #     idx = CArray.arange(0, self.tr.num_samples)
    #     val_dts_idx = CArray.randsample(idx, 500, random_state=self.seed)
    #     self._val = self.tr[val_dts_idx, :]
    #
    #     tr_dts_idx = CArray.randsample(idx, 1000, random_state=self.seed)
    #     self.tr = self.tr[tr_dts_idx, :]
    #
    #     idx = CArray.arange(0, self.ts.num_samples)
    #     ts_dts_idx = CArray.randsample(idx, 1000, random_state=self.seed)
    #     self.ts = self.ts[ts_dts_idx, :]
    #
    #     self.transform_train = transforms.Compose([
    #         transforms.Lambda(lambda x: x.reshape([1, 28, 28])),
    #     ])

    def _load_mnist(self):

        self.tr, self.val, self.ts, self.tr2 = create_mnist_dataset(
            seed=self.seed, digits=[8, 9], n_tr=500, n_val=1000, n_ts=1000)

        # define the trasformation that will be applied just before to feed
        # the samples to the neural network
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

        print("Labels:\n{:}".format(labels))
        print("Scores:\n{:}".format(scores))

        acc = CMetricAccuracy().performance_score(self.ts.Y, labels)

        return acc

    def _load_pois_data(self):

        if os.path.isfile(self.pois_data_path + '.npz'):
            with np.load(self.pois_data_path + '.npz') as fm:
                X = fm['X']
                Y = fm['Y']
            X = CArray(X)
            Y = CArray(Y)
            pois_data = CDataset(X, Y)[:self.n_pois_points, :]
            print "number of poisoning samples ", pois_data.num_samples
        else:
            raise ValueError("file not found")

        return pois_data

    def __init__(self):

        self.seed = 0

        self.n_pois_points = 25
        #self.pois_data_path = "/home/ambra/np_adv/mnist_0_logistic"
        #self.pois_data_path = "/home/ambra/np_adv_tr/mnist_0_logistic"
        self.pois_data_path = "/home/ambra/new_np_adv/secml_code"

        self._load_mnist()

        self.clf = CClassifierPyTorchCarliniCNNMNIST(num_classes=2,
                                                     random_state=0,
                                                     train_transform=self.transform_train)

    def test_pois_efficacy(self):

        print("Test the efficacy of the poisoning point")

        self.clf.fit(self.tr2)

        clear_acc = self._get_accuracy(self.clf)
        print("Accuracy of the classifier trained on the "
              "original training dataset: {:}".format(clear_acc))

        self.clf.clear()

        pois_data = self._load_pois_data()
        pois_dts = self.tr2.append(pois_data)
        print "pois dts X shape ", pois_dts.X.shape

        self.clf.fit(pois_dts)

        pois_acc = self._get_accuracy(self.clf)
        print("Accuracy of the classifier trained on the "
              "poisoned training dataset: {:}".format(pois_acc))


test = TestPoisoningPointEfficacy()
test.test_pois_efficacy()
