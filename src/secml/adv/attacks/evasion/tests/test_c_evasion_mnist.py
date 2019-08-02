from secml.testing import CUnitTest

from numpy import random

from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernel import CKernel
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.data.loader import CDataLoaderMNIST
from secml.array import CArray
from secml.figure import CFigure
from secml.utils import fm


class TestEvasionMNIST(CUnitTest):
    """Evasion on MNIST datset."""

    def _param_setter(self):

        self.x0_img_class = 1
        self.y_target = None

        self.sparse = False
        self.distance = 'l1'
        self.dmax = 10

        self.eta = 1.0 / 255.0
        self.eta_min = 0.1
        self.eta_max = None

        self.classifier = CClassifierMulticlassOVA(
            classifier=CClassifierSVM, C=1.0,
            kernel=CKernel.create('rbf', gamma=0.01),
        )

        self.surrogate_classifier = self.classifier

        self.lb = 0.0
        self.ub = 1.0

        self.filename = 'test_c_evasion_mnist.pdf'

    def setUp(self):

        self.seed = None

        if self.seed is None:
            self.seed = random.randint(999999999)

        self._param_setter()
        self._dataset_creation()
        self.classifier.fit(self._tr)
        self._choose_x0()

        # adversarial example creation
        self._evasion_obj = CAttackEvasionPGDLS(
            classifier=self.classifier,
            surrogate_classifier=self.classifier,
            surrogate_data=self._val_dts,
            distance=self.distance,
            attack_classes='all',
            lb=self.lb,
            ub=self.ub,
            dmax=self.dmax,
            y_target=self.y_target,
            solver_params={'eta': self.eta,
                           'eta_min': self.eta_min,
                           'eta_max': self.eta_max,
                           'eps': 1e-6})

        self._evasion_obj.verbose = 2

    def _dataset_creation(self):

        loader = CDataLoaderMNIST()

        n_tr = 500
        n_val = 100
        n_ts = 100

        self._digits = [4, 9]

        self._tr = loader.load(
            'training', digits=self._digits, num_samples=n_tr+n_val)
        self._ts = loader.load(
            'testing', digits=self._digits, num_samples=n_ts)

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
        val_dts_idx = CArray.randsample(idx, n_val, random_state=self.seed)
        self._val_dts = self._tr[val_dts_idx, :]

        tr_dts_idx = CArray.randsample(idx, n_tr, random_state=self.seed)
        self._tr = self._tr[tr_dts_idx, :]

        idx = CArray.arange(0, self._ts.num_samples)
        ts_dts_idx = CArray.randsample(idx, n_ts, random_state=self.seed)
        self._ts = self._ts[ts_dts_idx, :]

    def _choose_x0(self):
        """Find a sample of that belong to the required class."""
        adv_img_idx = \
            CArray(self._ts.Y.find(self._ts.Y == self.x0_img_class))[0]

        self._x0 = self._ts.X[adv_img_idx, :]
        self._y0 = self._ts.Y[adv_img_idx]

    def _show_adv(self, x0, y0, xopt, y_pred):
        """Show the original and the modified sample.

        Parameters
        ----------
        x0
            Original image.
        xopt
            Modified sample.

        """
        added_noise = abs(xopt - x0)  # absolute value of noise image

        fig = CFigure(height=5.0, width=15.0)
        fig.subplot(1, 3, 1)
        fig.sp.title(self._digits[y0.item()])
        fig.sp.imshow(x0.reshape((self._tr.header.img_h,
                                  self._tr.header.img_w)), cmap='gray')
        fig.subplot(1, 3, 2)
        fig.sp.imshow(
            added_noise.reshape((self._tr.header.img_h,
                                 self._tr.header.img_w)), cmap='gray')
        fig.subplot(1, 3, 3)
        fig.sp.title(self._digits[y_pred.item()])
        fig.sp.imshow(xopt.reshape((self._tr.header.img_h,
                                    self._tr.header.img_w)), cmap='gray')
        fig.savefig(
            fm.join(fm.abspath(__file__), self.filename), file_format='pdf')

    def test_evasion(self):
        y_pred, scores, p_opt = self._evasion_obj.run(self._x0, self._y0)[:3]
        self._show_adv(self._x0, self._y0, p_opt.X, y_pred[0])


if __name__ == '__main__':
    CUnitTest.main()
