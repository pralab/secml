import time
from abc import ABCMeta

from secml.adv.attacks.poisoning import CAttackPoisoningLogisticRegression, \
    CAttackPoisoningRidge, CAttackPoisoningSVM
from secml.adv.attacks.poisoning.tests import CAttackPoisoningLinTest
from secml.figure import CFigure
from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers import CClassifierSVM, CClassifierRidge, \
    CClassifierSGD
from secml.ml.peval.metrics import CMetric
from secml.optimization import COptimizer
from secml.optimization.constraints import CConstraintBox
from secml.optimization.function import CFunction
from secml.utils import CUnitTest


class CPoisoningTestCases(object):
    class TestCPoisoning(CUnitTest):
        """Unit test for CAttackPoisoning."""
        __metaclass__ = ABCMeta

        def _clf_creation(self):

            if self.clf_idx == 'logistic':

                self.classifier = CClassifierSGD(loss='log', regularizer='l2',
                                                 alpha=0.0001)

                self.pois_class = CAttackPoisoningLogisticRegression

                self.discr_f_level = 0

            elif self.clf_idx == 'svm':

                self.classifier = CClassifierSVM(kernel='linear', C=0.01)

                self.pois_class = CAttackPoisoningSVM
                self.discr_f_level = 0

            elif self.clf_idx == 'ridge':

                self.classifier = CClassifierRidge(fit_intercept=True,
                                                   alpha=1) #0.01

                self.pois_class = CAttackPoisoningRidge

                self.discr_f_level = 0
            else:
                raise ValueError("classifier idx not managed!")

        def _pois_obj_creation(self):

            # self.solver_type = 'gradient-descent'
            # self.solver_params = {'eta': 0.05, 'eps': 1e-9}

            self.solver_type = 'descent-direction'
            self.solver_params = {'eta': 0.05, 'eta_min': 0.1, 'eps': 1e-9}

            self._poisoning_params = {
                "classifier": self.classifier,
                "training_data": self.tr,
                "surrogate_classifier": self.classifier,
                "surrogate_data": self.tr,
                "ts": self.ts,
                "lb": self.lb,
                "ub": self.ub,
                "discrete": False,
                "solver_type": self.solver_type,
                "solver_params": self.solver_params,
                'random_seed': self.seed
            }

            self.poisoning = self.pois_class(**self._poisoning_params)
            self.poisoning.verbose = 2  # enables print on terminal
            self.poisoning.n_points = 1  # 1
            self.xc, self.yc = self.poisoning._rnd_init_poisoning_points()

            self.logger.info('yc: ' + str(self.yc))

            self.name_file = 'poisoning.pdf'

        def setUp(self):

            # Setting all defined parameter
            self.param_setter()
            self.verbose = 2

            self._dataset_creation()
            self._clf_creation()

            start_training = time.time()
            self.classifier.train(self.tr)
            self.clf_orig = self.classifier.deepcopy()
            end_training = time.time()
            print "training time ", end_training - start_training

            self._test_accuracy(self.classifier)

            self._pois_obj_creation()

            return

        #####################################################################
        #                             TESTED METHODS
        #####################################################################

        def grad_check(self, xc):
            # Compare analytical gradient with its numerical approximation
            check_grad_val = COptimizer(
                CFunction(self.poisoning._objective_function,
                          self.poisoning._objective_function_gradient)
            ).check_grad(xc)
            self.logger.info("Gradient difference between analytical svm "
                             "gradient and numerical gradient: %s",
                             str(check_grad_val))
            self.assertLess(check_grad_val, 1e-1,
                            "poisoning gradient is wrong {:}".format(
                                check_grad_val))
            for i, elm in enumerate(self.xc.size):
                self.assertIsInstance(elm, float)

        def _test_accuracy(self, clf):
            metric = CMetric.create('accuracy')
            y_pred, scores = clf.classify(self.ts.X)
            acc = metric.performance_score(y_true=self.ts.Y, y_pred=y_pred)
            self.logger.info("Error on testing data: " + str(1 - acc))


        def _clf_poisoning(self):

            print "self.yc before run ", self.yc

            # with self.logger.timer():
            xc = self.poisoning._run(self.xc, self.yc)

            self.logger.info("Starting score: " + str(self.poisoning.f_seq[0]))
            self.logger.info("Final score: " + str(self.poisoning.f_seq[-1]))
            self.logger.info("x*: " + str(xc))
            self.logger.info("Point sequence: " + str(self.poisoning.x_seq))
            self.logger.info("Score sequence: : " + str(self.poisoning.f_seq))
            self.logger.info("Fun Eval: " + str(self.poisoning.f_eval))
            self.logger.info("Grad Eval: " + str(self.poisoning.grad_eval))

            metric = CMetric.create('accuracy')
            y_pred, scores = self.classifier.classify(self.ts.X)
            acc = metric.performance_score(y_true=self.ts.Y, y_pred=y_pred)
            self.logger.info("Error on testing data: " + str(1 - acc))

            tr = self.tr.append(CDataset(xc, self.yc))

            pois_clf = self.classifier.deepcopy()
            pois_clf.clear()

            pois_clf.train(tr)
            y_pred, scores = pois_clf.classify(self.ts.X)
            acc = metric.performance_score(y_true=self.ts.Y, y_pred=y_pred)
            self.logger.info(
                "Error on testing data (poisoned): " + str(1 - acc))

            return pois_clf

        def test_poisoning_2D_plot(self):

            pois_clf = self._clf_poisoning()

            if self.n_features == 2:

                fig = CFigure(height=4, width=10)
                n_rows = 1
                n_cols = 2

                fig.subplot(n_rows, n_cols, grid_slot=1)
                fig.sp.title('Attacker objective and gradients')
                self._plot_func(fig, self.poisoning._objective_function)
                self._plot_obj_grads(
                    fig, self.poisoning._objective_function_gradient)
                self._plot_ds(fig, self.tr)
                self._plot_clf(fig, self.clf_orig, self.tr,
                               background=False, line_color='k')
                self._plot_clf(fig, pois_clf, self.tr, background=False)
                self._plot_box(fig)
                fig.sp.plot_path(self.poisoning.x_seq,
                                 start_facecolor='r' if self.yc == 1 else 'b')

                fig.subplot(n_rows, n_cols, grid_slot=2)
                fig.sp.title('Classification error on ts')
                self._plot_func(fig, self.poisoning._objective_function,
                                acc=True)
                self._plot_ds(fig, self.tr)
                self._plot_clf(fig, pois_clf, self.tr, background=False)
                self._plot_box(fig)
                fig.sp.plot_path(self.poisoning.x_seq,
                                 start_facecolor='r' if self.yc == 1 else 'b')

                fig.show()
                fig.savefig(self.name_file, file_format='pdf')

        #####################################################################
        #                             INTERNALS
        #####################################################################

        def _plot_param_sub(self, fig, param_fun, grad_fun, clf):

            self._plot_func(fig, param_fun)
            self._plot_obj_grads(
                fig, grad_fun)
            self._plot_ds(fig, self.tr)
            self._plot_clf(fig, clf, self.tr, background=False)
            self._plot_box(fig)

        def _plot_func(self, fig, func, **func_kwargs):
            """Plot poisoning objective function"""
            fig.switch_sptype(sp_type="function")
            fig.sp.plot_fobj(
                func=func,
                grid_limits=self.grid_limits, plot_levels=False,
                n_grid_points=20, colorbar=True, **func_kwargs)

        def _plot_obj_grads(self, fig, func, **func_kwargs):
            """Plot poisoning attacker objective function gradient"""
            fig.switch_sptype(sp_type="function")
            fig.sp.plot_fgrads(
                func,
                grid_limits=self.grid_limits,
                n_grid_points=20, **func_kwargs)

        def _plot_clf(self, fig, clf, ds, background=True, line_color='gray'):
            """Plot the decision function of a multiclass classifier."""

            def plot_hyperplane(img, clf, min_v, max_v, linestyle, label):
                """Plot the hyperplane associated to the OVA clf."""
                xx = CArray.linspace(
                    min_v - 5, max_v + 5)  # make sure the line is long enough
                # get the separating hyperplane
                yy = -(clf.w[0] * xx + clf.b) / clf.w[1]
                img.sp.plot(xx, yy, linestyle, label=label)

            # x_bounds, y_bounds = ds.get_bounds()
            x_bounds, y_bounds = self.grid_limits[0], self.grid_limits[1]

            if ds.num_classes == 3:
                styles = [('b', 'o', '-'), ('g', 'p', '--'), ('r', 's', '-.')]
            elif ds.num_classes == 4:
                styles = [('saddlebrown', 'o', '-'), ('g', 'p', '--'),
                          ('y', 's', '-.'), ('gray', 'D', '--')]
            else:
                styles = [('saddlebrown', 'o', '-'), ('g', 'p', '--'),
                          ('y', 's', '-.'), ('gray', 'D', '--'),
                          ('c', '-.'), ('m', '-'), ('y', '-.')]

            for c_idx, c in enumerate(ds.classes):
                # Plot boundary and predicted label for each OVA classifier

                fig.sp.scatter(ds.X[ds.Y == c, 0],
                               ds.X[ds.Y == c, 1],
                               s=70, c=styles[c_idx][0], edgecolors='k',
                               facecolors='none', linewidths=1,
                               )

            # Plotting multiclass decision function
            fig.switch_sptype('function')
            fig.sp.plot_fobj(lambda x: clf.classify(x)[0],
                             multipoint=True, cmap='Set2',
                             grid_limits=self.grid_limits,
                             colorbar=False, n_grid_points=100,
                             plot_levels=True,
                             plot_background=background, levels=[0, 1, 2],
                             levels_color=line_color, levels_style='--')

            fig.sp.xlim(x_bounds[0] - .05, x_bounds[1] + .05)
            fig.sp.ylim(y_bounds[0] - .05, y_bounds[1] + .05)

            fig.sp.legend(loc=9, ncol=3, mode="expand", handletextpad=.1)

            return fig

        def _plot_ds(self, fig, data):
            """Plot dataset"""
            fig.switch_sptype(sp_type="ds")
            fig.sp.plot_ds(data)

        def _plot_box(self, fig):
            """Plot box constraint"""
            fig.switch_sptype(sp_type="function")
            # construct and plot box
            if self.lb == "x0":
                self.lb = self.x0
            if self.ub == "x0":
                self.ub = self.x0
            box = CConstraintBox(lb=self.lb, ub=self.ub)
            fig.sp.plot_fobj(func=box.constraint,
                             plot_background=False,
                             n_grid_points=20,
                             grid_limits=self.grid_limits,
                             levels=[self.discr_f_level], colorbar=False)

        def test_poisoning_grad_check(self):

            pois_clf = self._clf_poisoning()

            xc = self.xc

            # Compare analytical gradient with its numerical approximation
            check_grad_val = COptimizer(
                CFunction(self.poisoning._objective_function,
                          self.poisoning._objective_function_gradient)
            ).check_grad(xc)
            self.logger.info("Gradient difference between analytical "
                             "poisoning "
                             "gradient and numerical gradient: %s",
                             str(check_grad_val))
            self.assertLess(check_grad_val, 1,
                            "poisoning gradient is wrong {:}".format(
                                check_grad_val))
