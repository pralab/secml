from secml.adv.attacks.evasion.tests import CAttackEvasionTestCases

from secml.adv.attacks.evasion import CAttackEvasionPGDExp

from secml.array import CArray


class TestCAttackEvasionPGDExp(CAttackEvasionTestCases):
    """Unittests for CAttackEvasionPGDExp."""

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
        if not params["classifier"].is_fitted():
            self.logger.info("Training classifier...")
            params["classifier"].fit(ds.X, ds.Y)

        evas = CAttackEvasionPGDExp(**params)
        evas.verbose = 2

        # pick a malicious sample
        x0, y0 = self._choose_x0_2c(ds)

        return evas, x0, y0

    def test_linear_l1(self):
        """Test evasion of a linear classifier using L1 distance."""

        eta = 0.01
        sparse = True
        seed = 10

        ds, clf = self._prepare_linear_svm(sparse, seed)

        evasion_params = {
            "classifier": clf,
            "double_init_ds": ds,
            "distance": 'l1',
            "dmax": 1.05,
            "lb": -1.05,
            "ub": 1.05,
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
        expected_x = CArray([0.0176, -1.05])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(evas, ds, x0, 'pgd_exp_linear_L1.pdf')

    def test_linear_l1_discrete(self):
        """Test evasion of a linear classifier using L1 distance (discrete)."""

        eta = 0.5
        sparse = True
        seed = 10

        ds, clf = self._prepare_linear_svm(sparse, seed)

        ds = self._discretize_data(ds, eta)

        evasion_params = {
            "classifier": clf,
            "double_init_ds": ds,
            "distance": 'l1',
            "dmax": 2,
            "lb": -1,
            "ub": 1,
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
        expected_x = CArray([0.5, -1])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(evas, ds, x0, 'pgd_exp_linear_L1_discrete.pdf')

    def test_linear_l1_discrete_10d(self):
        """Test evasion of a linear classifier (10 features)
        using L1 distance (discrete).
        In this test we set few features to the same value to cover a
        special case of the l1 projection, where there are multiple
        features with the same max value. The optimizer should change
        one of them at each iteration.
        """

        eta = 0.5
        sparse = True
        seed = 10

        ds, clf = self._prepare_linear_svm_10d(sparse, seed)

        ds = self._discretize_data(ds, eta)

        evasion_params = {
            "classifier": clf,
            "double_init_ds": ds,
            "distance": 'l1',
            "dmax": 5,
            "lb": -2,
            "ub": 2,
            "attack_classes": CArray([1]),
            "y_target": 0,
            "solver_params": {
                "eta": eta,
                "eta_min": None,
                "eta_max": None
            }
        }

        evas, x0, y0 = self._set_evasion(ds, evasion_params)

        # Set few features to the same max value
        w_new = clf.w.deepcopy()
        w_new[CArray.randint(
            clf.w.size, shape=3, random_state=seed)] = clf.w.max()
        clf._w = w_new

        # Expected final optimal point
        # CAttackEvasionPGDExp uses CLineSearchBisectProj
        # which brings the point outside of the grid
        expected_x = \
            CArray([-1.8333, -1.8333, 1.8333, 0, -0.5, 0, 0.5, -0.5, 1, 0.5])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

    def test_linear_l2(self):
        """Test evasion of a linear classifier using L2 distance."""

        eta = 0.5
        sparse = True
        seed = 48574308

        ds, clf = self._prepare_linear_svm(sparse, seed)

        evasion_params = {
            "classifier": clf,
            "double_init_ds": ds,
            "distance": 'l2',
            "dmax": 1.05,
            "lb": -0.67,
            "ub": 0.67,
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

        self._plot_2d_evasion(evas, ds, x0, 'pgd_exp_linear_L2.pdf')

    def test_nonlinear_l1(self):
        """Test evasion of a nonlinear classifier using L1 distance."""

        eta = 0.1
        sparse = False
        seed = 87985889

        ds, clf = self._prepare_nonlinear_svm(sparse, seed)

        evasion_params = {
            "classifier": clf,
            "double_init_ds": ds,
            "distance": 'l1',
            "dmax": 1.0,
            "lb": -1.0,
            "ub": 1.0,
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
        expected_x = CArray([-0.1384, -0.8484])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(evas, ds, x0, 'pgd_exp_nonlinear_L1.pdf')

    def test_nonlinear_l2(self):
        """Test evasion of a nonlinear classifier using L2 distance."""

        eta = 0.01
        sparse = False
        seed = 534513

        ds, clf = self._prepare_nonlinear_svm(sparse, seed)

        evasion_params = {
            "classifier": clf,
            "double_init_ds": ds,
            "distance": 'l2',
            "dmax": 1.25,
            "lb": -0.65,
            "ub": 1.0,
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
        expected_x = CArray([-0.5883, -0.4128])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(evas, ds, x0, 'pgd_exp_nonlinear_L2.pdf')

    def test_tree_l1(self):
        """Test evasion of a tree classifier using L1 distance."""

        eta = 1.0
        sparse = False
        seed = 0

        ds, clf, clf_surr = self._prepare_tree_nonlinear_svm(sparse, seed)

        evasion_params = {
            "classifier": clf_surr,
            "double_init_ds": ds,
            "distance": 'l1',
            "dmax": 2.0,
            "lb": -1.5,
            "ub": 1.5,
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
        expected_x = CArray([-1.3775, 1.3775])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(
            evas, ds, x0, th=0.5, filename='pgd_exp_tree_L1.pdf')
