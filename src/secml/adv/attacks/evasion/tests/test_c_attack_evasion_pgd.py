from secml.adv.attacks.evasion.tests import CAttackEvasionTestCases

from secml.adv.attacks.evasion import CAttackEvasionPGD

from secml.array import CArray


class TestCAttackEvasionPGD(CAttackEvasionTestCases):
    """Unittests for CAttackEvasionPGD."""

    def _set_evasion(self, ds, params):
        """Prepare the evasion attack.

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
            params["classifier"].fit(ds)

        if not params["surrogate_classifier"].is_fitted():
            self.logger.info("Training surrogate classifier...")
            params["surrogate_classifier"].fit(params["surrogate_data"])

        evas = CAttackEvasionPGD(**params)
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
            "surrogate_classifier": clf,
            "surrogate_data": ds,
            "distance": 'l1',
            "dmax": 1.05,
            "lb": -1.05,
            "ub": 1.05,
            "attack_classes": CArray([1]),
            "y_target": 0,
            "solver_params": {
                "eta": eta
            }
        }

        evas, x0, y0 = self._set_evasion(ds, evasion_params)

        # Expected final optimal point
        expected_x = CArray([0.0176, -1.05])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(evas, ds, x0, 'pdg_linear_L1.pdf')

    def test_linear_l2(self):
        """Test evasion of a linear classifier using L2 distance."""

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
            "attack_classes": CArray([1]),
            "y_target": 0,
            "solver_params": {
                "eta": eta
            }
        }

        evas, x0, y0 = self._set_evasion(ds, evasion_params)

        # Expected final optimal point
        expected_x = CArray([0.1697, 0.67])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(evas, ds, x0, 'pdg_linear_L2.pdf')

    def test_nonlinear_l1(self):
        """Test evasion of a nonlinear classifier using L1 distance."""

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
            "attack_classes": CArray([1]),
            "y_target": 0,
            "solver_params": {
                "eta": eta
            }
        }

        evas, x0, y0 = self._set_evasion(ds, evasion_params)

        # Expected final optimal point
        expected_x = CArray([-0.1726, -0.8141])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(evas, ds, x0, 'pdg_nonlinear_L1.pdf')

    def test_nonlinear_l2(self):
        """Test evasion of a nonlinear classifier using L2 distance."""

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
            "attack_classes": CArray([1]),
            "y_target": 0,
            "solver_params": {
                "eta": eta
            }
        }

        evas, x0, y0 = self._set_evasion(ds, evasion_params)

        # Expected final optimal point
        expected_x = CArray([-0.6096, -0.3796])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(evas, ds, x0, 'pdg_nonlinear_L2.pdf')

    def test_tree_l1(self):
        """Test evasion of a tree classifier using L1 distance."""

        eta = 0.1
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
            "attack_classes": CArray([1]),
            "y_target": 0,
            "solver_params": {
                "eta": eta
            }
        }

        evas, x0, y0 = self._set_evasion(ds, evasion_params)

        # Expected final optimal point
        expected_x = CArray([1.28, 0.3497])
        expected_y = 0

        self._run_evasion(evas, x0, y0, expected_x, expected_y)

        self._plot_2d_evasion(
            evas, ds, x0, th=0.5, filename='pdg_tree_L1.pdf')
