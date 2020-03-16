from secml.adv.attacks.evasion.tests import CAttackEvasionTestCases

from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernel
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.data.loader import CDataLoaderMNIST
from secml.array import CArray
from secml.figure import CFigure
from secml.utils import fm


class TestCAttackEvasionPGDLSMNIST(CAttackEvasionTestCases):
    """Unittests for CAttackEvasionPGDLS on MNIST dataset."""

    def _load_mnist49(self, sparse=False, seed=None):
        """Load MNIST49 dataset.

        - load dataset
        - normalize in 0-1
        - split in training (500), validation (100), test (100)

        Parameters
        ----------
        sparse : bool, optional (default False)
        seed : int or None, optional (default None)

        """
        loader = CDataLoaderMNIST()

        n_tr = 500
        n_val = 100
        n_ts = 100

        self._digits = [4, 9]

        self._tr = loader.load(
            'training', digits=self._digits, num_samples=n_tr+n_val)
        self._ts = loader.load(
            'testing', digits=self._digits, num_samples=n_ts)

        if sparse is True:
            self._tr = self._tr.tosparse()
            self._ts = self._ts.tosparse()

        # normalize in [lb,ub]
        self._tr.X /= 255.0
        self._ts.X /= 255.0

        idx = CArray.arange(0, self._tr.num_samples)
        val_dts_idx = CArray.randsample(idx, n_val, random_state=seed)
        self._val_dts = self._tr[val_dts_idx, :]

        tr_dts_idx = CArray.randsample(idx, n_tr, random_state=seed)
        self._tr = self._tr[tr_dts_idx, :]

        idx = CArray.arange(0, self._ts.num_samples)
        ts_dts_idx = CArray.randsample(idx, n_ts, random_state=seed)
        self._ts = self._ts[ts_dts_idx, :]

    def _set_evasion(self, params, x0_img_class):
        """Prepare the evasion attack.

        - train the classifier (if not trained)
        - train the surrogate classifier (if not trained)
        - create the evasion object
        - choose an attack starting point

        Parameters
        ----------
        params : dict
            Parameters for the attack class.
        x0_img_class : int
            Class from which to choose the initial attack point.

        Returns
        -------
        evas : CAttackEvasion
        x0 : CArray
            Initial attack point.
        y0 : CArray
            Label of the initial attack point.

        """
        if not params["classifier"].is_fitted():
            self.logger.info("Training classifier...")
            params["classifier"].fit(self._tr)

        if not params["surrogate_classifier"].is_fitted():
            self.logger.info("Training surrogate classifier...")
            params["surrogate_classifier"].fit(params["surrogate_data"])

        evas = CAttackEvasionPGDLS(**params)
        evas.verbose = 2

        # pick a malicious sample
        x0, y0 = self._choose_x0_2c(x0_img_class)

        return evas, x0, y0

    def _choose_x0_2c(self, x0_img_class):
        """Find a sample of that belong to the required class."""
        adv_img_idx = \
            CArray(self._ts.Y.find(self._ts.Y == x0_img_class))[0]

        x0 = self._ts.X[adv_img_idx, :]
        y0 = self._ts.Y[adv_img_idx]

        return x0, y0

    def _prepare_multiclass_svm(self, sparse, seed):
        """Preparare the data required for attacking a MULTICLASS SVM.

        - load the MNIST dataset
        - create a MULTICLASS SVM with RBF kernel (C=1, gamma=0.01)

        Parameters
        ----------
        sparse : bool
        seed : int or None

        Returns
        -------
        ds : CDataset
        clf : CClassifierSVM

        """
        self._load_mnist49(sparse, seed)

        clf = CClassifierMulticlassOVA(
            classifier=CClassifierSVM, C=1.0,
            kernel=CKernel.create('rbf', gamma=0.01),
        )

        return clf

    def test_mnist(self):
        """Test evasion of a multiclass classifier on MNIST dataset."""

        sparse = False
        seed = 128690268

        clf = self._prepare_multiclass_svm(sparse, seed)

        evasion_params = {
            "classifier": clf,
            "surrogate_classifier": clf,
            "surrogate_data": self._val_dts,
            "distance": 'l1',
            "dmax": 10,
            "lb": 0,
            "ub": 1,
            "attack_classes": 'all',
            "y_target": None,
            "solver_params": {
                "eta": 1.0 / 255.0,
                "eta_min": 0.1,
                "eta_max": None,
                "eps": 1e-6
            }
        }

        evas, x0, y0 = self._set_evasion(evasion_params, x0_img_class=1)

        self._run_evasion(evas, x0, y0, expected_y=0)  # label 0 is digit 4

        y_pred = evas.classifier.predict(evas.x_opt)

        self.filename = 'pgd_ls_mnist.pdf'
        self._show_adv(x0, y0, evas.x_opt, y_pred[0])

    def _show_adv(self, x0, y0, x_opt, y_pred):
        """Show the original and the modified sample.

        Parameters
        ----------
        x0 : CArray
            Initial attack point.
        y0 : CArray
            Label of the initial attack point.
        x_opt : CArray
            Final optimal point.
        y_pred : CArray
            Predicted label of the final optimal point.

        """
        if self.make_figures is False:
            self.logger.debug("Skipping figures...")
            return

        added_noise = abs(x_opt - x0)  # absolute value of noise image

        fig = CFigure(height=5.0, width=15.0)
        fig.subplot(1, 3, 1)
        fig.sp.title(self._digits[y0.item()])
        fig.sp.imshow(x0.reshape(
            (self._tr.header.img_h, self._tr.header.img_w)), cmap='gray')
        fig.subplot(1, 3, 2)
        fig.sp.imshow(
            added_noise.reshape(
                (self._tr.header.img_h, self._tr.header.img_w)), cmap='gray')
        fig.subplot(1, 3, 3)
        fig.sp.title(self._digits[y_pred.item()])
        fig.sp.imshow(x_opt.reshape(
            (self._tr.header.img_h, self._tr.header.img_w)), cmap='gray')
        fig.savefig(
            fm.join(self.images_folder, self.filename), file_format='pdf')


if __name__ == '__main__':
    CAttackEvasionTestCases.main()
