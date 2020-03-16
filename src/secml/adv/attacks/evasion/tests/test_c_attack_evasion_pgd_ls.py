from secml.adv.attacks.evasion.tests import CAttackEvasionTestCases

from secml.adv.attacks.evasion import CAttackEvasionPGDLS

from secml.array import CArray


class TestCAttackEvasionPGDLS(CAttackEvasionTestCases):
    """Unittests for CAttackEvasionPGDLS."""

    def _set_evasion(self, ds, params):
        """Prepare the evasion attack.

        - discretize data if necessary
        - train the classifier (if not trained)
        - train the surrogate classifier (if not trained)
        - create the evasion object
        - choose an attack starting point

        Parameters
        ----------
        ds : CDataset
        params : dict
            Parameters for the attack class.

        Returns
        -------
        evas : CAttackEvasion
        x0 : CArray
            Initial attack point.
        y0 : CArray
            Label of the initial attack point.

        """
        if params["discrete"] is True:
            ds = self._discretize_data(ds, params["solver_params"]["eta"])

        if not params["classifier"].is_fitted():
            self.logger.info("Training classifier...")
            params["classifier"].fit(ds)

        if not params["surrogate_classifier"].is_fitted():
            self.logger.info("Training surrogate classifier...")
            params["surrogate_classifier"].fit(params["surrogate_data"])

        evas = CAttackEvasionPGDLS(**params)
        evas.verbose = 2

        # pick a malicious sample
        x0, y0 = self._choose_x0_2c(ds)

        return evas, x0, y0

    def test_linear_l1(self):
        """Test evasion of a linear classifier using L1 distance."""

        discrete = True
        eta = 0.01
        sparse = True
        seed = 10

        ds, clf = self._prepare_linear_svm(sparse, seed)

        evasion_params = {
            "classifier": clf,
            "surrogate_classifier": clf,
            "surrogate_data": ds,
            "distance": 'l1',
            "dmax": 1.05,
            "lb": -1.05,
            "ub": 1.05,
            "discrete": discrete,
            "attack_classes": CArray([1]),
            "y_target": 0,
            "solver_params": {
                "eta": eta,
                "eta_min": None,
                "eta_max": None
            }
        }

        evas, x0, y0 = self._set_evasion(ds, evasion_params)

        # Expected final optimal point
        expected_x = CArray([0.02, -1.05])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(evas, ds, x0, 'pgd_ls_linear_L1.pdf')

    def test_linear_l2(self):
        """Test evasion of a linear classifier using L2 distance."""

        discrete = False
        eta = 0.5
        sparse = True
        seed = 48574308

        ds, clf = self._prepare_linear_svm(sparse, seed)

        evasion_params = {
            "classifier": clf,
            "surrogate_classifier": clf,
            "surrogate_data": ds,
            "distance": 'l2',
            "dmax": 1.05,
            "lb": -0.67,
            "ub": 0.67,
            "discrete": discrete,
            "attack_classes": CArray([1]),
            "y_target": 0,
            "solver_params": {
                "eta": eta,
                "eta_min": None,
                "eta_max": None
            }
        }

        evas, x0, y0 = self._set_evasion(ds, evasion_params)

        # Expected final optimal point
        expected_x = CArray([0.4463, 0.67])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(evas, ds, x0, 'pgd_ls_linear_L2.pdf')

    def test_nonlinear_l1(self):
        """Test evasion of a nonlinear classifier using L1 distance."""

        discrete = False
        eta = 0.1
        sparse = False
        seed = 87985889

        ds, clf = self._prepare_nonlinear_svm(sparse, seed)

        evasion_params = {
            "classifier": clf,
            "surrogate_classifier": clf,
            "surrogate_data": ds,
            "distance": 'l1',
            "dmax": 1.0,
            "lb": -1.0,
            "ub": 1.0,
            "discrete": discrete,
            "attack_classes": CArray([1]),
            "y_target": 0,
            "solver_params": {
                "eta": eta,
                "eta_min": 0.1,
                "eta_max": None
            }
        }

        evas, x0, y0 = self._set_evasion(ds, evasion_params)

        # Expected final optimal point
        expected_x = CArray([-0.19, -0.7967])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(evas, ds, x0, 'pgd_ls_nonlinear_L1.pdf')

    def test_nonlinear_l2(self):
        """Test evasion of a nonlinear classifier using L2 distance."""

        discrete = False
        eta = 0.01
        sparse = False
        seed = 534513

        ds, clf = self._prepare_nonlinear_svm(sparse, seed)

        evasion_params = {
            "classifier": clf,
            "surrogate_classifier": clf,
            "surrogate_data": ds,
            "distance": 'l2',
            "dmax": 1.25,
            "lb": -0.65,
            "ub": 1.0,
            "discrete": discrete,
            "attack_classes": CArray([1]),
            "y_target": 0,
            "solver_params": {
                "eta": eta,
                "eta_min": 0.01,
                "eta_max": None
            }
        }

        evas, x0, y0 = self._set_evasion(ds, evasion_params)

        # Expected final optimal point
        expected_x = CArray([-0.5975, -0.3988])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(evas, ds, x0, 'pgd_ls_nonlinear_L2.pdf')

    def test_tree_l1(self):
        """Test evasion of a tree classifier using L1 distance."""

        discrete = False
        eta = 1.0
        sparse = False
        seed = 0

        ds, clf, clf_surr = self._prepare_tree_nonlinear_svm(sparse, seed)

        evasion_params = {
            "classifier": clf,
            "surrogate_classifier": clf_surr,
            "surrogate_data": ds,
            "distance": 'l1',
            "dmax": 2.0,
            "lb": -1.5,
            "ub": 1.5,
            "discrete": discrete,
            "attack_classes": CArray([1]),
            "y_target": 0,
            "solver_params": {
                "eta": eta,
                "eta_min": None,
                "eta_max": None
            }
        }

        evas, x0, y0 = self._set_evasion(ds, evasion_params)

        # Expected final optimal point
        expected_x = CArray([-1.0988, 1.5])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(
            evas, ds, x0, th=0.5, filename='pgd_ls_tree_L1.pdf')
