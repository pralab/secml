from __future__ import division

from secml.testing import CUnitTest
from abc import ABCMeta, abstractmethod
import six
from six.moves import range

from numpy import *
import time

from secml.core.type_utils import is_list
from secml.array import CArray
from secml.data.loader import CDLRandomBlobs
from secml.figure import CFigure
from secml.utils import fm

from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.optim.constraints import \
    CConstraintBox, CConstraintL1, CConstraintL2


class CEvasionTestCases:
    """Wrapper for TestCEvasion to make unittest.main() work correctly."""

    @six.add_metaclass(ABCMeta)
    class TestCEvasion(CUnitTest):
        """Unit test for CEvasion."""

        @abstractmethod
        def param_setter(self):
            pass

        def setUp(self):

            # Setting all defined parameter
            self.param_setter()

            if self.seed is None:
                self.seed = random.randint(999999999)

            # use dense evasion by default
            if not hasattr(self, 'sparse'):
                self.sparse = False

            loader = CDLRandomBlobs(
                n_samples=50,
                n_features=self.n_features,
                centers=self.n_clusters,
                center_box=(-0.5, 0.5),
                cluster_std=0.5,
                random_state=self.seed)

            self.logger.info(
                "Loading `random_blobs` with seed: {:}".format(self.seed))
            self.dataset = loader.load()

            # discretize dataset on equi-spaced grid of step "eta"
            if self.discrete is True:
                self._discretize_data()

            self.logger.info("Initializing SVM with training data... ")

            self.classifier.fit(self.dataset)

            # pick a malicious sample and init evasion
            malicious_idxs = self.dataset.Y.find(self.dataset.Y == 1)
            target_idx = 1  # random.choice(range(0, len(malicious_idxs)))

            self.x0 = self.dataset.X[malicious_idxs[target_idx], :].ravel()
            self.y0 = +1

            self.logger.info("Malicious sample: " + str(self.x0))

            self.solver_params = {
                "eta": self.eta,
                "eta_min": self.eta_min,
                "eta_max": self.eta_max
            }

            if self.sparse is True:
                self.logger.info("Converting data to sparse...")
                self.dataset = self.dataset.tosparse()

            if not self.surrogate_classifier.is_fitted():
                self.surrogate_classifier.fit(self.dataset)

            params = {
                "classifier": self.classifier,
                "surrogate_classifier": self.surrogate_classifier,
                "surrogate_data": self.dataset,
                "distance": self.type_dist,
                "dmax": self.dmax,
                "lb": self.lb,
                "ub": self.ub,
                "discrete": self.discrete,
                "attack_classes": CArray([1]),
                "y_target": 0,
                "solver_params": self.solver_params
            }

            self.evasion = CAttackEvasionPGDLS(**params)
            self.evasion.verbose = 2

        # ####################################################################
        #                             TESTED METHODS
        # ####################################################################

        def _discretize_data(self):
            if is_list(self.eta):
                if len(self.eta) != self.n_features:
                    raise ValueError('len(eta) != n_features')
                for i in range(len(self.eta)):
                    self.dataset.X[:, i] = (
                        self.dataset.X[:, i] /
                        self.eta[i]).round() * self.eta[i]
            else:  # eta is a single value
                self.dataset.X = (self.dataset.X / self.eta).round() * self.eta

        def test_evasion(self):
            self.x_evas = self._run_evasion(self.evasion)

            # TODO: there should be a specific function for this
            # retrieve the full sequence of points
            self.x_evas = self.evasion.x_seq

            if self.n_features == 2:
                self._plot_evasion()

        # ####################################################################
        #                             INTERNALS
        # ####################################################################

        def _run_evasion(self, evas):
            """Test evasion method."""

            start_time = time.time()

            if self.sparse:
                y_pred, score, x = evas.run(self.x0.tosparse(), self.y0)[:3]
            else:
                y_pred, score, x = evas.run(self.x0, self.y0)[:3]
            self.logger.info("Is sparse?: " + str(x.issparse))
            if self.evasion._xk is not None:
                self.logger.info(
                    "Alternative init point(s):\n{:}".format(self.evasion._xk))
            final_time = time.time() - start_time
            self.logger.info("Starting score: " + str(
                self.classifier.decision_function(self.x0, y=1).item()))
            self.logger.info("Final score: " + str(evas.f_opt))
            self.logger.info("x*: " + str(evas.x_opt))
            self.logger.info("Point sequence: " + str(evas.x_seq))
            self.logger.info("Score sequence: : " + str(evas.f_seq))
            self.logger.info("Fun Eval: " + str(evas.f_eval))
            self.logger.info("Grad Eval: " + str(evas.grad_eval))
            self.logger.info("Evasion Time: " + str(final_time))
            return x

        def _plot_evasion(self):
            """Plot evasion results"""
            fig = CFigure(height=6, width=6)

            fig.subplot(n_rows=1, n_cols=1, grid_slot=1)
            self._plot_clf(self.evasion, fig, self.x_evas)
            # self._plot_clf_grad(fig)
            fig.savefig(fm.join(fm.abspath(__file__), self.filename), 
                        file_format='pdf')

        def _distance(self, x):
            """Rescale distance for plot"""
            c = self.x0
            if self.type_dist is "l1":
                constr = CConstraintL1(center=c, radius=self.dmax)
            else:
                constr = CConstraintL2(center=c, radius=self.dmax)
            return constr.constraint(x)

        def _plot_dataset_clf_and_box(self, evas, fig):
            """Plot dataset and box constraints"""
            fig.sp.plot_ds(self.dataset)
            fig.sp.plot_fun(
                func=evas.classifier.decision_function,
                grid_limits=self.grid_limits, colorbar=False,
                levels=[0], y=1)
            # construct and plot box
            if self.lb == "x0":
                self.lb = self.x0
            if self.ub == "x0":
                self.ub = self.x0
            box = CConstraintBox(lb=self.lb, ub=self.ub)
            fig.sp.plot_constraint(
                box, n_grid_points=50, grid_limits=self.grid_limits)

        def _plot_grid_and_path(self, fig, x, x_start):

            # plot distance constraint
            fig.sp.plot_fun(func=self._distance,
                            plot_background=False,
                            n_grid_points=50,
                            grid_limits=self.grid_limits,
                            levels=[0], colorbar=False)
            # plot optimization trace
            fig.sp.plot_path(self.x_evas)

        def _plot_clf_warm_start(self, fig, clf, x_start, x):
            """Plot classifier and path for multistep evasion"""
            self._plot_grid_and_path(fig, x, x_start)

        def _plot_clf(self, evas, fig, x):
            """Plot classifier for standard evasion"""
            self._plot_dataset_clf_and_box(evas, fig)
            self._plot_grid_and_path(fig, x, self.x0)

        def clf_grad(self, x):
            self.classifier.gradient_f_x(x)
            return

        def _plot_clf_grad(self, fig):
            """Plot classifier grad for standard evasion"""
            fig.sp.plot_fgrads(
                self.clf_grad,
                grid_limits=self.grid_limits,
                n_grid_points=20)
