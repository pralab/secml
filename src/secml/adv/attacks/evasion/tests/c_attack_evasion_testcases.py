from secml.testing import CUnitTest

from numpy import *

from secml.data.loader import CDLRandomBlobs
from secml.optim.constraints import \
    CConstraintBox, CConstraintL1, CConstraintL2
from secml.ml.features.normalization import CNormalizerMinMax
from secml.ml.classifiers import CClassifierSVM, CClassifierDecisionTree
from secml.core.type_utils import is_list, is_float
from secml.figure import CFigure
from secml.utils import fm

IMAGES_FOLDER = fm.join(fm.abspath(__file__), 'test_images')
if not fm.folder_exist(IMAGES_FOLDER):
    fm.make_folder(IMAGES_FOLDER)


class CAttackEvasionTestCases(CUnitTest):
    """Unittests interface for CAttackEvasion."""
    images_folder = IMAGES_FOLDER
    make_figures = False  # Set as True to produce figures

    def _load_blobs(self, n_feats, n_clusters, sparse=False, seed=None):
        """Load Random Blobs dataset.

        - n_samples = 50
        - center_box = (-0.5, 0.5)
        - cluster_std = 0.5

        Parameters
        ----------
        n_feats : int
        n_clusters : int
        sparse : bool, optional (default False)
        seed : int or None, optional (default None)

        """
        loader = CDLRandomBlobs(
            n_samples=50,
            n_features=n_feats,
            centers=n_clusters,
            center_box=(-0.5, 0.5),
            cluster_std=0.5,
            random_state=seed)

        self.logger.info(
            "Loading `random_blobs` with seed: {:}".format(seed))
        ds = loader.load()

        if sparse is True:
            ds = ds.tosparse()

        return ds

    @staticmethod
    def _discretize_data(ds, eta):
        """Discretize data of input dataset based on eta.

        Parameters
        ----------
        ds : CDataset
        eta : eta or scalar

        """
        if is_list(eta):
            if len(eta) != ds.n_features:
                raise ValueError('len(eta) != n_features')
            for i in range(len(eta)):
                ds.X[:, i] = (ds.X[:, i] / eta[i]).round() * eta[i]
        else:  # eta is a single value
            ds.X = (ds.X / eta).round() * eta

        return ds

    def _prepare_linear_svm(self, sparse, seed):
        """Preparare the data required for attacking a LINEAR SVM.

        - load a blob 2D dataset
        - create a SVM (C=1) and a minmax preprocessor

        Parameters
        ----------
        sparse : bool
        seed : int or None

        Returns
        -------
        ds : CDataset
        clf : CClassifierSVM

        """
        ds = self._load_blobs(
            n_feats=2,  # Number of dataset features
            n_clusters=2,  # Number of dataset clusters
            sparse=sparse,
            seed=seed
        )

        normalizer = CNormalizerMinMax(feature_range=(-1, 1))
        clf = CClassifierSVM(C=1.0, preprocess=normalizer)

        return ds, clf

    def _prepare_nonlinear_svm(self, sparse, seed):
        """Preparare the data required for attacking a NONLINEAR SVM.

        - load a blob 2D dataset
        - create a SVM with RBF kernel (C=1, gamma=1) and a minmax preprocessor

        Parameters
        ----------
        sparse : bool
        seed : int or None

        Returns
        -------
        ds : CDataset
        clf : CClassifierSVM

        """
        ds = self._load_blobs(
            n_feats=2,  # Number of dataset features
            n_clusters=2,  # Number of dataset clusters
            sparse=sparse,
            seed=seed
        )

        normalizer = CNormalizerMinMax(feature_range=(-1, 1))
        clf = CClassifierSVM(kernel='rbf', C=1, preprocess=normalizer)

        return ds, clf

    def _prepare_tree_nonlinear_svm(self, sparse, seed):
        """Preparare the data required for attacking a TREE classifier with
        surrogate NONLINEAR SVM.

        - load a blob 2D dataset
        - create a decision tree classifier
        - create a surrogate SVM with RBF kernel (C=1, gamma=1)

        Parameters
        ----------
        sparse : bool
        seed : int or None

        Returns
        -------
        ds : CDataset
        clf : CClassifierDecisionTree
        clf_surr : CClassifierSVM

        """
        ds = self._load_blobs(
            n_feats=2,  # Number of dataset features
            n_clusters=2,  # Number of dataset clusters
            sparse=sparse,
            seed=seed
        )

        clf = CClassifierDecisionTree(random_state=seed)
        clf_surr = CClassifierSVM(kernel='rbf', C=1)

        return ds, clf, clf_surr

    @staticmethod
    def _choose_x0_2c(ds):
        """Choose a starting point having label 1 from a 2-class ds.

        Parameters
        ----------
        ds : CDataset
            2-class dataset.

        Returns
        -------
        x0 : CArray
            Initial attack point.
        y0 : CArray
            Label of the initial attack point.

        """
        if ds.num_classes != 2:
            raise ValueError("Only 2-class datasets can be used!")

        malicious_idxs = ds.Y.find(ds.Y == 1)
        target_idx = 1

        x0 = ds.X[malicious_idxs[target_idx], :].ravel()
        y0 = +1

        return x0, y0

    def _run_evasion(self, evas, x0, y0, expected_x=None, expected_y=None):
        """Run evasion on input x.

        Parameters
        ----------
        evas : CAttackEvasion
        x0 : CArray
            Initial attack point.
        y0 : CArray
            Label of the initial attack point.
        expected_x : CArray or None, optional
            Expected final optimal point.
        expected_y : int or CArray or None, optional
            Label of the expected final optimal point.

        """
        self.logger.info("Malicious sample: " + str(x0))
        self.logger.info("Is sparse?: " + str(x0.issparse))

        with self.logger.timer():
            y_pred, scores, adv_ds, f_obj = evas.run(x0, y0)

        self.logger.info("Starting score: " + str(
            evas.classifier.decision_function(x0, y=1).item()))

        self.logger.info("Final score: " + str(evas.f_opt))
        self.logger.info("x*:\n" + str(evas.x_opt))
        self.logger.info("Point sequence:\n" + str(evas.x_seq))
        self.logger.info("Score sequence:\n" + str(evas.f_seq))
        self.logger.info("Fun Eval: " + str(evas.f_eval))
        self.logger.info("Grad Eval: " + str(evas.grad_eval))

        # Checking output
        self.assertEqual(1, y_pred.size)
        self.assertEqual(1, scores.shape[0])
        self.assertEqual(1, adv_ds.num_samples)
        self.assertEqual(adv_ds.issparse, x0.issparse)
        self.assertTrue(is_float(f_obj))

        # Compare optimal point with expected
        if expected_x is not None:
            self.assert_array_almost_equal(
                evas.x_opt.todense().ravel(), expected_x, decimal=4)
        if expected_y is not None:
            self.assert_array_almost_equal(y_pred.item(), expected_y)

    @staticmethod
    def _constr(evas, c):
        """Return the distance constraint depending on the used distance.

        Parameters
        ----------
        evas : CAttackEvasion

        Returns
        -------
        CConstraintL1 or CConstraintL2

        """
        # TODO: there is no way to cleanly extract it from evasion object
        if evas.distance is "l1":
            constr = CConstraintL1(center=c, radius=evas.dmax)
        else:
            constr = CConstraintL2(center=c, radius=evas.dmax)
        return constr

    @staticmethod
    def _box(evas):
        """Return the bounding box constraint.

        Parameters
        ----------
        evas : CAttackEvasion

        Returns
        -------
        CConstraintBox

        """
        # TODO: there is no way to cleanly extract it from evasion object
        return CConstraintBox(lb=evas.lb, ub=evas.ub)

    def _plot_2d_evasion(self, evas, ds, x0, filename, th=0, grid_limits=None):
        """Plot evasion attack results for 2D data.

        Parameters
        ----------
        evas : CAttackEvasion
        ds : CDataset
        x0 : CArray
            Initial attack point.
        filename : str
            Name of the output pdf file.
        th : scalar, optional
            Scores threshold of the classifier. Default 0.
        grid_limits : list of tuple or None, optional
            If not specified, will be set as [(-1.5, 1.5), (-1.5, 1.5)].

        """
        if self.make_figures is False:
            self.logger.debug("Skipping figures...")
            return

        fig = CFigure(height=6, width=6)

        if grid_limits is None:
            grid_limits = [(-1.5, 1.5), (-1.5, 1.5)]

        fig.sp.plot_ds(ds)
        fig.sp.plot_fun(
            func=evas.classifier.decision_function,
            grid_limits=grid_limits, colorbar=False,
            n_grid_points=50, levels=[th], y=1)

        fig.sp.plot_constraint(self._box(evas),
                               n_grid_points=20,
                               grid_limits=grid_limits)

        fig.sp.plot_fun(func=lambda z: self._constr(evas, x0).constraint(z),
                        plot_background=False,
                        n_grid_points=50,
                        grid_limits=grid_limits,
                        levels=[0],
                        colorbar=False)

        fig.sp.plot_path(evas.x_seq)

        fig.savefig(fm.join(self.images_folder, filename), file_format='pdf')

