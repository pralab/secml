from secml.adv.attacks.evasion.tests import CAttackEvasionTestCases

from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.kernels import CKernelRBF
from secml.utils import fm

from secml.figure import CFigure
from secml.optim.constraints import CConstraintL2
from secml.ml.features.normalization import CNormalizerMinMax

from secml.adv.attacks.evasion import CAttackEvasionPGDLS


# TODO: COMBINE WITH TestCAttackEvasionPGDLSMNIST
class TestCAttackEvasionPGDLSRejectThreshold(CAttackEvasionTestCases):
    """Unittests for CAttackEvasionPGDLS on with Reject Threshold clf."""

    def setUp(self):

        import numpy as np
        np.random.seed(12345678)

        self._dataset_creation()

        self.kernel = CKernelRBF(gamma=1)

        self.multiclass = CClassifierMulticlassOVA(
            classifier=CClassifierSVM, class_weight='balanced',
            preprocess=None, kernel=self.kernel)
        self.multiclass.verbose = 0

        self.multiclass = CClassifierRejectThreshold(self.multiclass, 0.6)

        # Training and classification
        self.multiclass.fit(self.ds)

        self.y_pred, self.score_pred = self.multiclass.predict(
            self.ds.X, return_decision_function=True)

    def _dataset_creation(self):
        # generate synthetic data
        self.ds = CDLRandom(n_samples=100, n_classes=3, n_features=2,
                            n_redundant=0, n_clusters_per_class=1,
                            class_sep=1, random_state=0).load()

        # Add a new class modifying one of the existing clusters
        self.ds.Y[(self.ds.X[:, 0] > 0).logical_and(
            self.ds.X[:, 1] > 1).ravel()] = self.ds.num_classes

        self.lb = 0
        self.ub = 1

        # Data normalization
        self.normalizer = CNormalizerMinMax(
            feature_range=(self.lb, self.ub))
        self.normalizer = None
        if self.normalizer is not None:
            self.ds.X = self.normalizer.fit_transform(self.ds.X)

    def test_indiscriminate(self):
        """Test indiscriminate evasion."""

        self.y_target = None
        self.logger.info("Test indiscriminate evasion ")

        expected_x = CArray([0.8952, 0.1099])
        self._test_evasion_multiclass(expected_x)

    def test_targeted(self):
        """Test targeted evasion."""

        self.y_target = 2
        self.logger.info("Test target evasion "
                         "(with target class {:}) ".format(self.y_target))

        expected_x = CArray([2.3414, -0.5295])
        self._test_evasion_multiclass(expected_x)

    def _test_evasion_multiclass(self, expected_x):

        # EVASION
        self.multiclass.verbose = 2

        if self.normalizer is not None:
            lb = self.normalizer.feature_range[0]
            ub = self.normalizer.feature_range[1]
        else:
            lb = None
            ub = None

        dmax = 3

        self.solver_params = {'eta': 0.5, 'max_iter': 3}

        eva = CAttackEvasionPGDLS(classifier=self.multiclass,
                                  surrogate_classifier=self.multiclass,
                                  surrogate_data=self.ds,
                                  distance='l2', dmax=dmax, lb=lb, ub=ub,
                                  solver_params=self.solver_params,
                                  y_target=self.y_target)

        eva.verbose = 0  # 2

        # Points from class 2 region
        # p_idx = 0

        # Points from class 1 region
        # p_idx = 68

        # Points from class 3 region
        p_idx = 1  # Wrong classified point
        # p_idx = 53  # Evasion goes up usually

        # Points from class 0 region
        # p_idx = 49  # Wrong classified point
        # p_idx = 27  # Correctly classified point

        x0 = self.ds.X[p_idx, :]
        y0 = self.ds.Y[p_idx].item()

        self.logger.info("Evasion at dmax: " + str(dmax))

        eva.dmax = dmax
        x_opt, f_opt = eva._run(x0=x0, y0=y0, x_init=x0)
        y_pred, score = self.multiclass.predict(
            x_opt, return_decision_function=True)

        s = score[:, y0 if self.y_target is None else self.y_target]

        self.logger.info(
            "Number of objective function evaluations: {:}".format(
                eva.f_eval))

        self.logger.info(
            "Number of gradient function evaluations: {:}".format(
                eva.grad_eval))

        self.logger.info(
            "Predicted label after evasion: {:}".format(y_pred))
        self.logger.info("Score after evasion: {:}".format(s))
        self.logger.info(
            "Objective function after evasion: {:}".format(f_opt))

        # Compare optimal point with expected
        self.assert_array_almost_equal(
            eva.x_opt.todense().ravel(), expected_x, decimal=4)

        if self.y_target:

            s_ytarget_x0 = self.multiclass.decision_function(
                x0, self.y_target)
            s_ytarget_xopt = self.multiclass.decision_function(
                x_opt, self.y_target)

            self.logger.info(
                "Discriminat function w.r.t the target class first: {:} "
                "and after evasion: {:}".format(s_ytarget_x0,
                                                s_ytarget_xopt))

            self.assertLess(s_ytarget_x0, s_ytarget_xopt)

        else:  # indiscriminate attack

            s_ytrue_x0 = self.multiclass.decision_function(
                x0, y0)
            s_ytrue_xopt = self.multiclass.decision_function(
                x_opt, y0)

            self.logger.info(
                "Discriminat function w.r.t the true class first: {:} "
                "and after evasion: {:}".format(s_ytrue_x0,
                                                s_ytrue_xopt))

            self.assertGreater(s_ytrue_x0, s_ytrue_xopt)

        self._make_plot(p_idx, eva, dmax)

    def _make_plot(self, p_idx, eva, dmax):

        if self.make_figures is False:
            self.logger.debug("Skipping figures...")
            return

        x0 = self.ds.X[p_idx, :]
        y0 = self.ds.Y[p_idx].item()

        x_seq = CArray.empty((0, x0.shape[1]))
        scores = CArray([])
        f_seq = CArray([])

        x = x0
        for d_idx, d in enumerate(range(0, dmax + 1)):

            self.logger.info("Evasion at dmax: " + str(d))

            eva.dmax = d
            x, f_opt = eva._run(x0=x0, y0=y0, x_init=x)
            y_pred, score = self.multiclass.predict(
                x, return_decision_function=True)
            f_seq = f_seq.append(f_opt)
            # not considering all iterations, just values at dmax
            # for all iterations, you should bring eva.x_seq and eva.f_seq
            x_seq = x_seq.append(x, axis=0)

            s = score[:, y0 if self.y_target is None else self.y_target]

            scores = scores.append(s)

        self.logger.info(
            "Predicted label after evasion: {:}".format(y_pred))
        self.logger.info("Score after evasion: {:}".format(s))
        self.logger.info(
            "Objective function after evasion: {:}".format(f_opt))

        fig = CFigure(height=9, width=10, markersize=6, fontsize=12)

        # Get plot bounds, taking into account ds and evaded point path
        bounds_x, bounds_y = self.ds.get_bounds()
        min_x, max_x = bounds_x
        min_y, max_y = bounds_y
        min_x = min(min_x, x_seq[:, 0].min())
        max_x = max(max_x, x_seq[:, 0].max())
        min_y = min(min_y, x_seq[:, 1].min())
        max_y = max(max_y, x_seq[:, 1].max())
        ds_bounds = [(min_x, max_x), (min_y, max_y)]

        # Plotting multiclass decision regions
        fig.subplot(2, 2, 1)
        fig = self._plot_decision_function(fig, plot_background=True)

        fig.sp.plot_path(x_seq, path_style='-',
                         start_style='o', start_facecolor='w',
                         start_edgewidth=2, final_style='o',
                         final_facecolor='k', final_edgewidth=2)

        # plot distance constraint
        fig.sp.plot_fun(func=self._rescaled_distance,
                        multipoint=True,
                        plot_background=False,
                        n_grid_points=20, levels_color='k',
                        grid_limits=ds_bounds,
                        levels=[0], colorbar=False,
                        levels_linewidth=2.0, levels_style=':',
                        alpha_levels=.4, c=x0, r=dmax)

        fig.sp.grid(linestyle='--', alpha=.5, zorder=0)

        # Plotting multiclass evasion objective function
        fig.subplot(2, 2, 2)

        fig = self._plot_decision_function(fig)

        fig.sp.plot_fgrads(eva._objective_function_gradient,
                           grid_limits=ds_bounds, n_grid_points=20,
                           color='k', alpha=.5)

        fig.sp.plot_path(x_seq, path_style='-',
                         start_style='o', start_facecolor='w',
                         start_edgewidth=2, final_style='o',
                         final_facecolor='k', final_edgewidth=2)

        # plot distance constraint
        fig.sp.plot_fun(func=self._rescaled_distance,
                        multipoint=True,
                        plot_background=False,
                        n_grid_points=20, levels_color='w',
                        grid_limits=ds_bounds,
                        levels=[0], colorbar=False,
                        levels_style=':', levels_linewidth=2.0,
                        alpha_levels=.5, c=x0, r=dmax)

        fig.sp.plot_fun(lambda z: eva._objective_function(z),
                        multipoint=True,
                        grid_limits=ds_bounds,
                        colorbar=False, n_grid_points=20,
                        plot_levels=False)

        fig.sp.grid(linestyle='--', alpha=.5, zorder=0)

        fig.subplot(2, 2, 3)
        if self.y_target is not None:
            fig.sp.title("Classifier Score for Target Class (Targ. Evasion)")
        else:
            fig.sp.title("Classifier Score for True Class (Indiscr. Evasion)")
        fig.sp.plot(scores)

        fig.sp.grid()
        fig.sp.xlim(0, dmax)
        fig.sp.xlabel("dmax")

        fig.subplot(2, 2, 4)
        fig.sp.title("Objective Function")
        fig.sp.plot(f_seq)

        fig.sp.grid()
        fig.sp.xlim(0, dmax)
        fig.sp.xlabel("dmax")

        fig.tight_layout()

        k_name = self.kernel.class_type if self.kernel is not None else 'lin'
        fig.savefig(fm.join(
            self.images_folder,
            "pgd_ls_reject_threshold_{:}c_kernel-{:}_target-{:}.pdf".format(
                self.ds.num_classes, k_name, self.y_target)
        ))

    def _rescaled_distance(self, x, c, r):
        """Rescale distance for plot."""
        if self.normalizer is not None:
            c = self.normalizer.inverse_transform(c)
            x = self.normalizer.inverse_transform(x)
        constr = CConstraintL2(center=c, radius=r)
        return x.apply_along_axis(constr.constraint, axis=1)

    def _get_style(self):
        """Define the style vector for the different classes."""
        if self.ds.num_classes == 3:
            styles = [('b', 'o', '-'), ('g', 'p', '--'), ('r', 's', '-.')]
        elif self.ds.num_classes == 4:
            styles = [('saddlebrown', 'o', '-'), ('g', 'p', '--'),
                      ('y', 's', '-.'), ('gray', 'D', '--')]
        else:
            styles = [('saddlebrown', 'o', '-'), ('g', 'p', '--'),
                      ('y', 's', '-.'), ('gray', 'D', '--'),
                      ('c', '-.'), ('m', '-'), ('y', '-.')]

        return styles

    def _plot_decision_function(self, fig, plot_background=False):
        """Plot the decision function of a multiclass classifier."""
        fig.sp.title('{:}'.format(self.multiclass.__class__.__name__))

        x_bounds, y_bounds = self.ds.get_bounds()

        styles = self._get_style()

        for c_idx, c in enumerate(self.ds.classes):
            fig.sp.scatter(self.ds.X[self.ds.Y == c, 0],
                           self.ds.X[self.ds.Y == c, 1],
                           s=20, c=styles[c_idx][0], edgecolors='k',
                           facecolors='none', linewidths=1,
                           label='c {:}'.format(c))

        # Plotting multiclass decision function
        fig.sp.plot_fun(
            lambda x: self.multiclass.predict(x),
            multipoint=True, cmap='Set2',
            grid_limits=self.ds.get_bounds(offset=5),
            colorbar=False, n_grid_points=300,
            plot_levels=True, plot_background=plot_background,
            levels=[-1, 0, 1, 2], levels_color='gray', levels_style='--')

        fig.sp.xlim(x_bounds[0] - .05, x_bounds[1] + .05)
        fig.sp.ylim(y_bounds[0] - .05, y_bounds[1] + .05)

        fig.sp.legend(loc=9, ncol=5, mode="expand", handletextpad=.1)

        return fig


if __name__ == '__main__':
    CAttackEvasionTestCases.main()
