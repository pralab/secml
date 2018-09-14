import unittest
from prlib.utils import CUnitTest

from prlib.data.splitter import CDataSplitterKFold
from prlib.data.loader import CDLRandom, CDLRandomBlobs
from prlib.kernel import CKernelRBF

from prlib.classifiers.multiclass import CClassifierMulticlassOVA
from prlib.kernel import gamma_estimation

from prlib.classifiers import CClassifierSVM


class TestGammaSelection(CUnitTest):

    def _plot_decision_function(self, fig, multiclass, ds_test, y_pred):
        """Plot the decision function of a multiclass classifier."""

        fig.sp.title('{:} ({:})'.format(multiclass.__class__.__name__,
                                        multiclass.classifier.__name__))

        x_bounds, y_bounds = ds_test.get_bounds()

        if ds_test.num_classes == 3:
            styles = ['bo-', 'gp--', 'rs-.']
        elif ds_test.num_classes == 4:
            styles = ['bo-', 'cp--', 'ys-.', 'rD--']
        else:
            styles = ['bo-', 'cp--', 'ys-.', 'rD--', 'c-.', 'm-', 'y-.']

        for c_idx, c in enumerate(ds_test.classes):
            # Plot boundary and predicted label for each OVA classifier

            fig.sp.scatter(ds_test.X[ds_test.Y == c, 0],
                           ds_test.X[ds_test.Y == c, 1],
                           s=40, c=styles[c_idx][0])
            fig.sp.scatter(ds_test.X[y_pred == c, 0],
                           ds_test.X[y_pred == c, 1], s=160,
                           edgecolors=styles[c_idx][0],
                           facecolors='none', linewidths=2,
                           label='class {:}'.format(c))

        # Plotting multiclass decision function
        fig.switch_sptype('function')
        fig.sp.plot_fobj(lambda x: multiclass.classify(x)[0],
                         grid_limits=ds_test.get_bounds(offset=5),
                         colorbar=False, n_grid_points=20, plot_levels=True,
                         plot_background=False, levels=[-1, 0, 1, 2],
                         levels_color='gray', levels_style='--')

        fig.sp.xlim(x_bounds[0] - .5 * abs(x_bounds[0]),
                    x_bounds[1] + .5 * abs(x_bounds[1]))
        fig.sp.ylim(y_bounds[0] - .5 * abs(y_bounds[0]),
                    y_bounds[1] + .5 * abs(y_bounds[1]))

        fig.sp.legend(loc=2)  # upper, left

        return fig

    def test_gamma_estimation(self):
        # generate synthetic data
        self.ds = CDLRandomBlobs(n_samples=500, n_features=2,
                                 centers=3, cluster_std=1,
                                 random_state=1).load()

        splitter = CDataSplitterKFold(num_folds=2, random_state=1)
        splitter.compute_indices(self.ds)

        self.ds_train = self.ds[splitter.tr_idx[0], :]
        self.ds_test = self.ds[splitter.ts_idx[0], :]

        gamma = gamma_estimation(self.ds_train, factor=0.3)

        self.logger.info("gamma: {:}".format(gamma))

        clf_params = {'classifier': CClassifierSVM, 'C': 1.0,
                      'kernel': CKernelRBF(gamma=gamma), 'normalizer': None}

        multiclass = CClassifierMulticlassOVA(**clf_params)
        multiclass.verbose = 0

        # estimate best classifier parameters

        # Training the classifiers
        multiclass.train(self.ds_train)

        # another dataset classification
        y_pred, score_pred = multiclass.classify(self.ds_test.X)

        from prlib.figure import CFigure
        fig = CFigure()

        fig = self._plot_decision_function(
            fig, multiclass, self.ds_test, y_pred)

        fig.savefig('multiclass_rbf_gamma_estimation.pdf')


if __name__ == '__main__':
    unittest.main()
