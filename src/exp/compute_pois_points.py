import os

import numpy as np
import torchvision.transforms as transforms

from secml.ml.classifiers import CClassifierLogistic
from secml.pytorch.classifiers import CClassifierPyTorchCarliniCNNMNIST
from secml.core import CCreator
from secml.ml.peval.metrics import CMetricAccuracy
from secml.adv.attacks.poisoning import CAttackPoisoningLogisticRegression
from secml.adv.seceval import CSecEval
from dataset_creation import create_mnist_dataset


class TestComputePoisoningPoints(CCreator):
    """
    Test that compute a set of poisoning points.
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
            self.ts[50:100, :].X, return_decision_function=True)

        print("Labels:\n{:}".format(labels))
        print("Scores:\n{:}".format(scores))

        acc = CMetricAccuracy().performance_score(self.ts[50:100, :].Y, labels)

        return acc

    def _save_pois_data(self, pois_data):

        # store the poisoning points in numpy array
        np.savez_compressed(self.pois_data_path, X=pois_data.X.tondarray(),
                            Y=pois_data.Y.tondarray())

        if os.path.isfile(self.pois_data_path + '.npz'):
            with np.load(self.pois_data_path + '.npz') as fm:
                X = fm['X']
                Y = fm['Y']
        else:
            print("The poisoning points has not been correctly saved!")

    def __init__(self):

        self.seed = 0

        #self.pois_data_path = "/home/ambra/new_np_adv/secml_code"
        self.pois_data_path = "/home/ambra/new_np_adv/noinv_solver"

        self._load_mnist()

        self.surr_clf = CClassifierLogistic(C=1.0, random_seed=0)

        self.target_clf = CClassifierPyTorchCarliniCNNMNIST(num_classes=2,
                                                            random_state=0,
                                                            train_transform=self.transform_train)

    def compute_pois_points(self):

        print("Compute the poisoning points ")

        self.surr_clf.fit(self.tr)

        acc = self._get_accuracy(self.surr_clf)
        print("Accuracy of the classifier trained on the "
              "original training dataset: {:}".format(acc))

        # nb: the gradient-descent solver in lib does not have the grid search
        self.solver_type = 'gradient-descent'
        self.grad_desc_solver_params = {'eta': 0.1,  # [0.05, 0.1, 0.2, 0.5],
                                        'eps': 1e-6,
                                        'max_iter': 2}

        self.pois = CAttackPoisoningLogisticRegression(self.surr_clf,
                                                       training_data=self.tr,
                                                       surrogate_classifier=self.surr_clf,
                                                       ts=self.val,
                                                       surrogate_data=self.tr,
                                                       distance='l2',
                                                       dmax=1e20,
                                                       lb=0,
                                                       ub=1,
                                                       discrete=False,
                                                       y_target=None,
                                                       attack_classes='all',
                                                       solver_type=self.solver_type,
                                                       solver_params=self.grad_desc_solver_params,
                                                       init_type='random',
                                                       random_seed=0)

        self.param_name = 'n_points'

        self.param_values = [0, 5, 10, 15, 20, 26, 55, 125]
        self.sec_eval = CSecEval(self.pois, param_name=self.param_name,
                                 param_values=self.param_values,
                                 save_adv_ds=True)

        self.sec_eval.run_sec_eval(self.ts)

        self._save_pois_data(self.sec_eval.sec_eval_data.adv_ds[5])
        self._save_pois_data(self.sec_eval.sec_eval_data.adv_ds[7])

test = TestComputePoisoningPoints()
test.compute_pois_points()

#2019-01-28 20:11:19,708 - CAttackPoisoningLogisticRegression.0x7f98159e7f50
# - INFO - Original classifier accuracy on test data 0.848
#2019-01-28 20:11:19,709 - CSecEval.0x7f98261c3f50 - INFO - Time: CArray([
# 200.034727])