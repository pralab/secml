from secml.utils import CUnitTest

from six.moves import range
import matplotlib

from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.kernel import CKernelRBF
from secml.utils import fm

from secml.figure import CFigure
from secml.optim.constraints import CConstraintL2
from secml.ml.features.normalization import CNormalizerMinMax

from secml.adv.attacks.evasion import CAttackEvasion


class TestEvasionMulticlass(CUnitTest):

    def setUp(self):

        self.show_plot = False

        import numpy as np
        np.random.seed(12345678)

        # generate synthetic data
        self.ds = CDLRandom(n_classes=3, n_features=2, n_redundant=0,
                            n_clusters_per_class=1, class_sep=1,
                            random_state=0).load()
        # self.ds = CDLRandomBlobs(
        #     n_samples=600, n_features=2, cluster_std=0.075, random_state=1,
        #     centers=[(0.25, 0.25), (0.5, 0.75), (0.85, 0.25)]).load()

        # Add a new class modifying one of the existing clusters
        self.ds.Y[(self.ds.X[:, 0] > 0).logical_and(
            self.ds.X[:, 1] > 1).ravel()] = self.ds.num_classes

        # self.kernel = None
        self.kernel = CKernelRBF(gamma=1)

        # END SETUP

        # Data normalization
        self.normalizer = CNormalizerMinMax()
        self.normalizer = None
        if self.normalizer is not None:
            self.ds.X = self.normalizer.fit_transform(self.ds.X)

        self.multiclass = CClassifierMulticlassOVA(
            classifier=CClassifierSVM, class_weight='balanced',
            preprocess=None, kernel=self.kernel)
        self.multiclass.verbose = 0

        # Training and classification
        self.multiclass.fit(self.ds)

        self.y_pred, self.score_pred = self.multiclass.predict(
            self.ds.X, return_decision_function=True)

    def test_evasion_indiscriminate(self):

        # set targeted (0,... c-1) or indiscriminate (None) evasion
        self.y_target = None

        self._test_evasion_multiclass()

    def test_evasion_targeted(self):

        # set targeted (0,... c-1) or indiscriminate (None) evasion
        self.y_target = 2

        self._test_evasion_multiclass()

    def _test_evasion_multiclass(self):

        # EVASION
        self.multiclass.verbose = 0

        if self.normalizer is not None:
            lb = self.normalizer.feature_range[0]
            ub = self.normalizer.feature_range[1]
        else:
            lb = None
            ub = None

        dmax = 4

        self.solver_type = 'descent-direction'
        self.solver_params = {'eta': 1e-1, 'eta_min': 0.1}

        eva = CAttackEvasion(classifier=self.multiclass,
                             surrogate_classifier=self.multiclass,
                             surrogate_data=self.ds,
                             distance='l2', dmax=dmax, lb=lb, ub=ub,
                             solver_type=self.solver_type,
                             solver_params=self.solver_params,
                             y_target=self.y_target)

        eva.verbose = 2

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
        for d_idx, d in enumerate(range(0, dmax + 1)):

            self.logger.info("Evasion at dmax: " + str(d))

            eva.dmax = d
            x, f_opt = eva._run(x0=x0, y0=y0, x_init=x)
            y_pred, score = self.multiclass.predict(
                x, return_decision_function=True)
            f_seq = f_seq.append(f_opt)
            # not considering all iterations, just values at dmax
            # for all iterations, you should bring eva.x_seq and eva.f_seq
            if x_seq is None:
                x_seq = x.deepcopy()
            else:
                x_seq = x_seq.append(x, axis=0)

            s = score[:, y0 if self.y_target is None else self.y_target]

            scores = scores.append(s)

        self.logger.info("Predicted label after evasion: {:}".format(y_pred))
        self.logger.info("Score after evasion: {:}".format(s))
        self.logger.info("Objective function after evasion: {:}".format(f_opt))

        # PLOT SECTION
        if self.show_plot:
            self._make_plots(x_seq, dmax, eva, x0, y0, scores, f_seq)

    def _make_plots(self, x_seq, dmax, eva, x0, y0, scores, f_seq):

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

        styles = self._get_style()

        fig.sp.plot_path(x_seq, path_style='-', path_width=2.5,
                         start_style='o', start_facecolor=styles[y0][0],
                         start_edgecolor='k',
                         start_edgewidth=1.5, final_style='D',
                         final_facecolor='r', final_edgecolor='k',
                         final_edgewidth=1.7)

        # plot distance constraint
        fig.sp.plot_fobj(func=self._rescaled_distance,
                         multipoint=True,
                         plot_background=False,
                         n_grid_points=100, levels_color='k',
                         grid_limits=ds_bounds,
                         levels=[0], colorbar=False,
                         levels_linewidth=2.0, levels_style=':',
                         alpha_levels=.4, c=x0, r=dmax)

        fig.sp.grid(grid_on=False)

        # Plotting multiclass evasion objective function
        fig.subplot(2, 2, 2)
        fig = self._plot_decision_function(fig)

        fig.switch_sptype('function')

        # # Use the actual target used in evasion
        # target = self.ds.Y[p_idx] if target_class is None else target_class
        fig.sp.plot_fgrads(eva._objective_function_gradient,
                           grid_limits=ds_bounds, n_grid_points=50,
                           color='k', alpha=.5)

        fig.sp.plot_path(x_seq, path_style='-', path_width=2.5,
                         start_style='o', start_facecolor=styles[y0][0],
                         start_edgecolor='k',
                         start_edgewidth=1.5, final_style='D',
                         final_facecolor='r', final_edgecolor='k',
                         final_edgewidth=1.7)

        # plot distance constraint
        fig.sp.plot_fobj(func=self._rescaled_distance,
                         multipoint=True,
                         plot_background=False,
                         n_grid_points=100, levels_color='w',
                         grid_limits=ds_bounds,
                         levels=[0], colorbar=False,
                         levels_style=':', levels_linewidth=2.0,
                         alpha_levels=.5, c=x0, r=dmax)

        fig.sp.plot_fobj(lambda x: eva._objective_function(x),
                         multipoint=True,
                         grid_limits=ds_bounds,
                         colorbar=False, n_grid_points=100,
                         plot_levels=False)

        fig.sp.grid(grid_on=False)

        fig.subplot(2, 2, 3)
        if self.y_target is not None:
            fig.sp.title("Classifier Score for Target Class (Targ. Evasion)")
        else:
            fig.sp.title("Classifier Score for True Class (Indiscr. Evasion)")
        fig.sp.plot(scores)

        fig.sp.grid()
        fig.sp.xticks(CArray.arange(dmax+1))
        fig.sp.xlim(0, dmax)
        fig.sp.xlabel("dmax")

        fig.subplot(2, 2, 4)
        fig.sp.title("Objective Function")
        fig.sp.plot(f_seq)

        fig.sp.grid()
        fig.sp.xticks(CArray.arange(dmax+1))
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

        self.logger.info("The plot has been shown")

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

    def _get_style(self):
        """
        This function define the style vector for the different classes
        Returns
        -------

        """
        if self.ds.num_classes == 3:
            styles = [('b', 'o', '-'), ('g', 'p', '--'), ('r', 's', '-.')]
        elif self.ds.num_classes == 4:
            styles = [('b', 'o', '-','cornflowerblue'), ('r', 'p', '--',
                                                     'lightcoral'),
                      ('y', 's', '-.', 'lemonchiffon'), ('g', 'D',
                                                                 '--',
                                                     'lightgreen')]
        else:
            styles = [('saddlebrown', 'o', '-'), ('g', 'p', '--'),
                      ('y', 's', '-.'), ('gray', 'D', '--'),
                      ('c', '-.'), ('m', '-'), ('y', '-.')]

        return styles

    def _plot_decision_function(self, fig):
        """Plot the decision function of a multiclass classifier."""

        fig.sp.title('{:} ({:})'.format(self.multiclass.__class__.__name__,
                                        self.multiclass.classifier.__name__))

        x_bounds, y_bounds = self.ds.get_bounds()

        styles = self._get_style()

        for c_idx, c in enumerate(self.ds.classes):

            fig.sp.scatter(self.ds.X[self.ds.Y == c, 0],
                           self.ds.X[self.ds.Y == c, 1],
                           s=20, c=styles[c_idx][0], edgecolors='k',
                           facecolors='none', linewidths=1,
                           label='c {:}'.format(c))

        # Plotting multiclass decision function
        fig.switch_sptype('function')

        colors = [style[3] for style in styles]
        # TODO: IMPLEMENT THIS IN CFIGURE
        cmap = matplotlib.colors.ListedColormap(
            colors, name='from_list', N=None)

        fig.sp.plot_fobj(lambda x: self.multiclass.predict(x),
                         multipoint=True, cmap=cmap,
                         grid_limits=self.ds.get_bounds(offset=5),
                         colorbar=False, n_grid_points=300, plot_levels=True,
                         plot_background=True, levels=[-1, 0, 1, 2],
                         levels_color='k', levels_style='-', alpha=.9,
                         levels_linewidth=0.9)

        fig.sp.xlim(x_bounds[0] - .05, x_bounds[1] + .05)
        fig.sp.ylim(y_bounds[0] - .05, y_bounds[1] + .05)

        fig.sp.legend(loc=9, ncol=5, mode="expand", handletextpad=.1)

        return fig


if __name__ == '__main__':
    CUnitTest.main()
