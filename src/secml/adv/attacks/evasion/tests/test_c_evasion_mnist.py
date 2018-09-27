from secml.classifiers import CClassifierSVM
from secml.kernel import CKernel
from secml.adv.attacks.evasion import CAttackEvasion
from secml.core import CCreator
from secml.data.loader import CDataLoaderMNIST
from secml.array import CArray
from secml.figure import CFigure
from secml.classifiers.multiclass import CClassifierMulticlassOVA

from numpy import random


class TestEvasionMNIST(CCreator):
    """Evasion on MNIST datset."""

    def _param_setter(self):

        self.x0_img_class = 1
        self.y_target = None

        self.sparse = False
        self.distance = 'l2'
        self.dmax = 2

        self.eta = 1.0 / 255.0
        self.eta_min = 0.1
        self.eta_max = None

        self.classifier = CClassifierMulticlassOVA(
            classifier=CClassifierSVM, C=1.0,
            kernel=CKernel.create('rbf', gamma=0.01),
        )

        self.lb = 0.0
        self.ub = 1.0

        self.name_file = 'MNIST_evasion.pdf'

    def __init__(self):

        self.seed = None

        if self.seed is None:
            self.seed = random.randint(999999999)

        print "seed: ", str(self.seed)

        self._param_setter()
        self._dataset_creation()
        print "training classifier ..."
        self.classifier.train(self._tr)
        print "training classifier ... Done."
        self._chose_x0()

        # adversarial example creation
        self._evasion_obj = CAttackEvasion(
            classifier=self.classifier,
            surrogate_classifier=self.classifier,
            surrogate_data=self._val_dts,
            distance=self.distance,
            attack_classes='all',
            lb=self.lb,
            ub=self.ub,
            dmax=self.dmax,
            y_target=self.y_target,
            solver_type='descent-direction',
            solver_params={'eta': self.eta,
                           'eta_min': self.eta_min,
                           'eta_max': self.eta_max,
                           'eps': 1e-6})

        self._evasion_obj.verbose = 2

    def _dataset_creation(self):
        loader = CDataLoaderMNIST()

        self._digits = CArray.arange(10).tolist()
        self._tr = loader.load('training', digits=self._digits)

        print "classes: ", self._tr.classes

        self._ts = loader.load('testing', digits=self._digits)

        # get dataset img dimension
        # TODO: these should not be included in CDataset!
        # they're lost after conversion tosparse
        self.img_w = self._tr.img_w
        self.img_h = self._tr.img_h

        if self.sparse is True:
            self._tr = self._tr.tosparse()
            self._ts = self._ts.tosparse()

        # normalize in [lb,ub]
        self._tr.X /= 255.0
        self._tr.X *= (self.ub - self.lb)
        if self.lb != 0:
            self._tr.X += self.lb
        self._ts.X /= 255.0
        self._ts.X *= (self.ub - self.lb)
        if self.lb != 0:
            self._ts.X += self.lb

        idx = CArray.arange(0, self._tr.num_samples)
        val_dts_idx = CArray.randsample(idx, 1000, random_state=self.seed)
        self._val_dts = self._tr[val_dts_idx, :]

        tr_dts_idx = CArray.randsample(idx, 5000, random_state=self.seed)
        self._tr = self._tr[tr_dts_idx, :]

        idx = CArray.arange(0, self._ts.num_samples)
        ts_dts_idx = CArray.randsample(idx, 100, random_state=self.seed)
        self._ts = self._ts[ts_dts_idx, :]

    def _chose_x0(self):
        """
        Find a sample of that belong to the required class
        :return:
        """
        adv_img_idx = \
            CArray(self._ts.Y.find(self._ts.Y == self.x0_img_class))[0]

        print "adv img idx ", adv_img_idx
        self._x0 = self._ts.X[adv_img_idx, :]
        self._y0 = self._ts.Y[adv_img_idx]

    def _show_adv(self, x0, y0, xopt, y_pred):
        """
        Show the original and the modified sample
        :param x0: original image
        :param xopt: modified sample
        """
        added_noise = abs(xopt - x0)  # absolute value of noise image

        if self.distance == 'l1':
            print "Norm of input perturbation (l1): ", \
                added_noise.ravel().norm(order=1)
        else:
            print "Norm of input perturbation (l2): ", \
                added_noise.ravel().norm()

        fig = CFigure(height=5.0, width=15.0)
        fig.subplot(1, 3, 1)
        fig.sp.title(self._digits[y0])
        fig.sp.imshow(x0.reshape((self.img_h, self.img_w)), cmap='gray')
        fig.subplot(1, 3, 2)
        fig.sp.imshow(
            added_noise.reshape((self.img_h, self.img_w)), cmap='gray')
        fig.subplot(1, 3, 3)
        fig.sp.title(self._digits[y_pred])
        fig.sp.imshow(xopt.reshape((self.img_h, self.img_w)), cmap='gray')
        fig.savefig(self.name_file, file_format='pdf')
        fig.show()

    def run(self):
        print "Run..."
        y_pred, scores, p_opt = self._evasion_obj.run(self._x0, self._y0)[:3]
        self._show_adv(self._x0, self._y0, p_opt.X, y_pred)


if __name__ == '__main__':
    TestEvasionMNIST().run()
