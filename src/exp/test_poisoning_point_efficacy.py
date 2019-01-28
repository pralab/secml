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
            # pois_data = CDataset(X, Y)[:self.n_pois_points, :]
            # pois_data = CDataset(X, Y)[-1:, :] #acc  0.979
            pois_data = CDataset(X, Y)[0, :]  # il punto calcolato con l'svm
            # la butta a 0.5
            print "number of poisoning samples ", pois_data.num_samples

            print "pois data min ", pois_data.X.min(axis=None)
            print "pois data max ", pois_data.X.max(axis=None)
            print "pois data label ", pois_data.Y
            pois_data.X[pois_data.X<0] = 0
            print "pois data min ", pois_data.X.min(axis=None)

            print "pois data x ", pois_data.X

            idx = CArray.arange(0, 784)
            from secml.figure import CFigure
            fig = CFigure()
            fig.subplot(1,2,1)
            fig.sp.title("svm pois point")
            fig.sp.bar(idx, pois_data.X.ravel())
            print pois_data.X.shape
            print type(pois_data.X[0,0])

            print "tr data min ", self.tr.X.min(axis=None)
            print "tr data max ", self.tr.X.max(axis=None)
            print "tr data mean sum ", self.tr.X.sum(axis = 0).mean(axis=None)

            print self.tr.X.mean(axis=0).shape
            fig.subplot(1,2,2)
            fig.sp.title("mean dataset point")
            fig.sp.bar(idx, self.tr.X.mean(axis=0).ravel())

            fig.show()

        else:
            raise ValueError("file not found")

        return pois_data

    def __init__(self):

        self.seed = 0

        self.n_pois_points = 1
        # computed on the validation dataset:
        #self.pois_data_path = "/home/ambra/np_adv/mnist_0_logistic"
        # self.pois_data_path = "/home/ambra/np_adv/mnist_0_ridge-10"

        # computed on the training dataset:
        #self.pois_data_path = "/home/ambra/np_adv_tr/mnist_0_logistic"
        # self.pois_data_path = "/home/ambra/np_adv_tr/mnist_0_ridge-10"

        self.pois_data_path = "/home/ambra/np_adv_tr/mnist_0_lin-svm-c100"

        #self.pois_data_path = "/home/ambra/new_np_adv/secml_code"
        # self.pois_data_path = "/home/ambra/new_np_adv/noinv_solver"

        self._load_mnist()

        # random state 0 acc tr single point svm 0.5 (partendo da 0.976)
        # random state 2 acc tr single point svm 0.974 (partendo da 0.977)

        self.clf = CClassifierPyTorchCarliniCNNMNIST(num_classes=2,
                                                     random_state=0, #random
                                                     # state 2 quasi
                                                     # ineffettivo un punto
                                                     # solo
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
