from abc import ABCMeta, abstractmethod
import six
from six.moves import range

from secml.utils import CUnitTest

from secml import _NoValue
from secml.adv.attacks.evasion import CAttackEvasion
from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.figure import CFigure
from secml.ml.features.normalization import CNormalizerMinMax
from secml.optim.constraints import CConstraintL2
from secml.utils import fm


class CEvasionRejectTestCases(object):
    """Wrapper for TestCEvasion to make unittest.main() work correctly."""

    @six.add_metaclass(ABCMeta)
    class TestCEvasionReject(CUnitTest):
        """Unit test for CEvasion."""

        @abstractmethod
        def _classifier_creation(self):
            raise NotImplementedError()

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
            self.normalizer = CNormalizerMinMax(feature_range=(self.lb,
                                                               self.ub))
            self.normalizer = None
            if self.normalizer is not None:
                self.ds.X = self.normalizer.fit_transform(self.ds.X)

        def setUp(self):

            self.show_plot = False

            import numpy as np
            np.random.seed(12345678)

            # END SETUP

            self._dataset_creation()
            self._classifier_creation()

            # Training and classification
            self.multiclass.fit(self.ds)

            self.y_pred, self.score_pred = self.multiclass.predict(
                self.ds.X, return_decision_function=True, n_jobs=_NoValue)

        def test_indiscriminate_evasion(self):

            self.logger.info(
                "Test indiscriminate evasion ")
            # set targeted (0,... c-1) or indiscriminate (None) evasion
            self.y_target = None
            self._test_evasion_multiclass()

        def test_target_evasion(self):

            # set targeted (0,... c-1) or indiscriminate (None) evasion
            self.y_target = 2
            self.logger.info(
                "Test target evasion (with target class {:}) ".format(
                    self.y_target))
            self._test_evasion_multiclass()

        def _test_evasion_multiclass(self):

            # EVASION
            self.multiclass.verbose = 2

            if self.normalizer is not None:
                lb = self.normalizer.feature_range[0]
                ub = self.normalizer.feature_range[1]
            else:
                lb = None
                ub = None

            dmax = 5

            # self.solver_type = 'descent-direction'
            # self.solver_params = {'eta': 1e-1, 'eta_min': 0.1}

            self.solver_type = 'gradient-descent'
            self.solver_params = {'eta': 0.5, 'max_iter': 3}

            eva = CAttackEvasion(classifier=self.multiclass,
                                 surrogate_classifier=self.multiclass,
                                 surrogate_data=self.ds,
                                 distance='l2', dmax=dmax, lb=lb, ub=ub,
                                 solver_type=self.solver_type,
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

            x_seq = None  # TODO: append from empty 2d arrays not supported
            scores = CArray([])
            f_seq = CArray([])

            x = x0

            self.logger.info("Evasion at dmax: " + str(dmax))

            eva.dmax = dmax
            x, f_opt = eva._run(x0=x0, y0=y0, x_init=x)
            y_pred, score = self.multiclass.predict(
                x, return_decision_function=True, n_jobs=_NoValue)
            f_seq = f_seq.append(f_opt)
            # not considering all iterations, just values at dmax
            # for all iterations, you should bring eva.x_seq and eva.f_seq
            if x_seq is None:
                x_seq = x.deepcopy()
            else:
                x_seq = x_seq.append(x, axis=0)

            s = score[:, y0 if self.y_target is None else self.y_target]

            scores = scores.append(s)

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

            x_opt = x_seq

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

            # PLOT SECTION
            if self.show_plot:
                self._make_plot(p_idx, eva, dmax)

        def _make_plot(self, p_idx, eva, dmax):

            x0 = self.ds.X[p_idx, :]
            y0 = self.ds.Y[p_idx].item()

            x_seq = None  # TODO: append from empty 2d arrays not supported
            scores = CArray([])
            f_seq = CArray([])

            x = x0
            for d_idx, d in enumerate(range(0, dmax + 1)):

                self.logger.info("Evasion at dmax: " + str(d))

                eva.dmax = d
                x, f_opt = eva._run(x0=x0, y0=y0, x_init=x)
                y_pred, score = self.multiclass.predict(
                    x, return_decision_function=True, n_jobs=_NoValue)
                f_seq = f_seq.append(f_opt)
                # not considering all iterations, just values at dmax
                # for all iterations, you should bring eva.x_seq and eva.f_seq
                if x_seq is None:
                    x_seq = x.deepcopy()
                else:
                    x_seq = x_seq.append(x, axis=0)

                s = score[:, y0 if self.y_target is None else self.y_target]

                scores = scores.append(s)

            self.logger.info(
                "Predicted label after evasion: {:}".format(y_pred))
            self.logger.info("Score after evasion: {:}".format(s))
            self.logger.info(
                "Objective function after evasion: {:}".format(f_opt))

            x_opt = x_seq[-1, :]

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
            fig = self._plot_decision_function(fig)

            fig.sp.plot_path(x_seq, path_style='-',
                             start_style='o', start_facecolor='w',
                             start_edgewidth=2, final_style='o',
                             final_facecolor='k', final_edgewidth=2)

            # plot distance constraint
            # for d_idx, d in enumerate([dmax]):
            for d in range(1, dmax + 1):
                fig.sp.plot_fobj(func=self._rescaled_distance,
                                 multipoint=True,
                                 plot_background=False,
                                 n_grid_points=20, levels_color='gray',
                                 grid_limits=ds_bounds,
                                 levels=[0], colorbar=False,
                                 levels_style=':',
                                 alpha_levels=.4, c=x0, r=d)

            fig.sp.grid(linestyle='--', alpha=.5, zorder=0)

            # Plotting multiclass evasion objective function
            fig.subplot(2, 2, 2)

            fig.switch_sptype('function')

            fig = self._plot_decision_function(fig)

            fig.sp.plot_fobj(lambda x: eva._objective_function(x),
                             multipoint=True,
                             grid_limits=ds_bounds,
                             colorbar=False, n_grid_points=20,
                             plot_levels=False)

            fig.sp.plot_fgrads(eva._objective_function_gradient,
                               grid_limits=ds_bounds, n_grid_points=20,
                               color='k', alpha=.5)

            fig.sp.plot_path(x_seq, path_style='-',
                             start_style='o', start_facecolor='w',
                             start_edgewidth=2, final_style='o',
                             final_facecolor='k', final_edgewidth=2)

            # plot distance constraint
            for d in range(1, dmax + 1):
                fig.sp.plot_fobj(func=self._rescaled_distance,
                                 multipoint=True,
                                 plot_background=False,
                                 n_grid_points=20, levels_color='w',
                                 grid_limits=ds_bounds,
                                 levels=[0], colorbar=False,
                                 levels_style=':',
                                 alpha_levels=.5, c=x0, r=d)

            fig.sp.grid(linestyle='--', alpha=.5, zorder=0)

            fig.subplot(2, 2, 3)
            if self.y_target is not None:
                fig.sp.title(
                    "Classifier Score for Target Class (Targ. Evasion)")
            else:
                fig.sp.title(
                    "Classifier Score for True Class (Indiscr. Evasion)")
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
            fig.show()

            k_name = self.kernel.class_type if self.kernel is not None else 'lin'
            fig.savefig(fm.join(
                fm.abspath(__file__),
                "multiclass_{:}c_kernel-{:}_target-{:}.pdf".format(
                    self.ds.num_classes, k_name, self.y_target)
            ))

            self.logger.info("The figure has been shown")

        #######################################
        # PRIVATE METHODS
        #######################################

        def _rescaled_distance(self, x, c, r):
            """Rescale distance for plot."""
            if self.normalizer is not None:
                c = self.normalizer.revert(c)
                x = self.normalizer.revert(x)
            constr = CConstraintL2(center=c, radius=r)
            return constr.constraint(x)

        def _plot_decision_function(self, fig):
            """Plot the decision function of a multiclass classifier."""

            def plot_hyperplane(img, clf, min_v, max_v, linestyle, label):
                """Plot the hyperplane associated to the OVA clf."""
                xx = CArray.linspace(
                    min_v - 5, max_v + 5)  # make sure the line is long enough
                # get the separating hyperplane
                yy = -(clf.w[0] * xx + clf.b) / clf.w[1]
                img.sp.plot(xx, yy, linestyle, label=label)

            fig.sp.title('{:}'.format(self.multiclass.__class__.__name__))

            x_bounds, y_bounds = self.ds.get_bounds()

            if self.ds.num_classes == 3:
                styles = [('b', 'o', '-'), ('g', 'p', '--'), ('r', 's', '-.')]
            elif self.ds.num_classes == 4:
                styles = [('saddlebrown', 'o', '-'), ('g', 'p', '--'),
                          ('y', 's', '-.'), ('gray', 'D', '--')]
            else:
                styles = [('saddlebrown', 'o', '-'), ('g', 'p', '--'),
                          ('y', 's', '-.'), ('gray', 'D', '--'),
                          ('c', '-.'), ('m', '-'), ('y', '-.')]

            for c_idx, c in enumerate(self.ds.classes):
                # Plot boundary and predicted label for each OVA classifier

                # plot_hyperplane(fig, self.multiclass.trained_classifiers[c_idx],
                #                 x_bounds[0], x_bounds[1], styles[c_idx],
                #                 'Boundary class {:}'.format(c))

                fig.sp.scatter(self.ds.X[self.ds.Y == c, 0],
                               self.ds.X[self.ds.Y == c, 1],
                               s=70, c=styles[c_idx][0], edgecolors='k',
                               facecolors='none', linewidths=1,
                               label='c {:}'.format(c))

            # Plotting multiclass decision function
            fig.switch_sptype('function')
            fig.sp.plot_fobj(
                lambda x: self.multiclass.predict(x, n_jobs=_NoValue),
                multipoint=True, cmap='Set2',
                grid_limits=self.ds.get_bounds(offset=5),
                colorbar=False, n_grid_points=200,
                plot_levels=True,
                plot_background=True, levels=[-1, 0, 1, 2],
                levels_color='gray', levels_style='--')

            fig.sp.xlim(x_bounds[0] - .05, x_bounds[1] + .05)
            fig.sp.ylim(y_bounds[0] - .05, y_bounds[1] + .05)

            fig.sp.legend(loc=9, ncol=5, mode="expand", handletextpad=.1)

            return fig
