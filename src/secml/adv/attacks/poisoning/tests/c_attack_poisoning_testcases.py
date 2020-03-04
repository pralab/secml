from secml.core import CCreator
from secml.data import CDataset
from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CDataSplitterShuffle
from secml.figure import CFigure
from secml.ml.features.normalization import CNormalizerMinMax
from secml.ml.peval.metrics import CMetric
from secml.optim.constraints import CConstraintBox
from secml.optim.function import CFunction
from secml.testing import CUnitTest


class _CAttackPoisoningLinTest(CCreator):
    """
    Debugging class for poisoning against linear classifiers.

    This class implement different functions which are useful to test
    the poisoning gradients of a linear classifier.
    """

    def __init__(self, pois_obj):
        self.pois_obj = pois_obj

    def w1(self, xc):
        """
        Parameters
        ----------
        xc: poisoning point

        Returns
        -------
        f_obj: values of objective function (average hinge loss) at x
        """
        idx, clf, tr = self._clf_poisoning(xc)

        return clf.w.ravel()[0]

    def w2(self, xc):
        """
        Parameters
        ----------
        xc: poisoning point

        Returns
        -------
        f_obj: values of objective function (average hinge loss) at x
        """

        idx, clf, tr = self._clf_poisoning(xc)

        return clf.w.ravel()[1]

    def b(self, xc):
        """
        Parameters
        ----------
        xc: poisoning point

        Returns
        -------
        f_obj: values of objective function (average hinge loss) at x
        """

        idx, clf, tr = self._clf_poisoning(xc)

        return clf.b

    def _clf_poisoning(self, xc):

        xc = xc.atleast_2d()
        n_samples = xc.shape[0]

        if n_samples > 1:
            raise TypeError("x is not a single sample!")

        # index of poisoning point within xc.
        # This will be replaced by the input parameter xc
        if self.pois_obj._idx is None:
            idx = 0
        else:
            idx = self.pois_obj._idx

        self.pois_obj._xc[idx, :] = xc
        clf, tr = self.pois_obj._update_poisoned_clf()

        return idx, clf, tr

    def _preparation_for_grad_computation(self, xc):

        idx, clf, tr = self._clf_poisoning(xc)

        y_ts = self.pois_obj._y_target if self.pois_obj._y_target is not \
                                          None else self.pois_obj.val.Y

        # computing gradient of loss(y, f(x)) w.r.t. f
        score = clf.decision_function(self.pois_obj.val.X)
        loss_grad = self.pois_obj._attacker_loss.dloss(y_ts, score)

        return idx, clf, loss_grad, tr

    def _grads_computation(self, xc):
        """
        Compute the derivative of the classifier parameters w.r.t. the
        poisoning points xc.

        The result is a CArray of dimension d * (d+1) where d is equal to the
        number of features

        """
        idx, clf, loss_grad, tr = self._preparation_for_grad_computation(xc)
        self.pois_obj._gradient_fk_xc(self.pois_obj._xc[idx, :],
                                      self.pois_obj._yc[idx],
                                      clf, loss_grad, tr)
        grads = self.pois_obj._d_params_xc
        return grads

    def gradient_w1_xc(self, xc):

        grad = self._grads_computation(xc)

        norm = grad.norm_2d()
        grad = grad / norm if norm > 0 else grad

        return grad[:, 0].ravel()

    def gradient_w2_xc(self, xc):

        grad = self._grads_computation(xc)

        norm = grad.norm_2d()
        grad = grad / norm if norm > 0 else grad

        return grad[:, 1].ravel()

    def gradient_b_xc(self, xc):

        grad = self._grads_computation(xc)

        norm = grad.norm_2d()
        grad = grad / norm if norm > 0 else grad

        return grad[:, 2].ravel()


class CAttackPoisoningTestCases(CUnitTest):
    def _dataset_creation(self):
        """Creates a blob dataset. """
        self.n_features = 2  # Number of dataset features

        self.seed = 42

        self.n_tr = 50
        self.n_ts = 100
        self.n_classes = 2

        loader = CDLRandomBlobs(
            n_samples=self.n_tr + self.n_ts,
            n_features=self.n_features,
            centers=[(-1, -1), (+1, +1)],
            center_box=(-2, 2),
            cluster_std=0.8,
            random_state=self.seed)

        self.logger.info(
            "Loading `random_blobs` with seed: {:}".format(self.seed))

        dataset = loader.load()
        splitter = CDataSplitterShuffle(num_folds=1, train_size=self.n_tr,
                                        random_state=3)
        splitter.compute_indices(dataset)
        self.tr = dataset[splitter.tr_idx[0], :]
        self.ts = dataset[splitter.ts_idx[0], :]

        normalizer = CNormalizerMinMax(feature_range=(-1, 1))
        self.tr.X = normalizer.fit_transform(self.tr.X)
        self.ts.X = normalizer.transform(self.ts.X)

        self.lb = -1
        self.ub = 1

        self.grid_limits = [(self.lb - 0.1, self.ub + 0.1),
                            (self.lb - 0.1, self.ub + 0.1)]

    def _create_poisoning_object(self):
        self.solver_type = 'pgd-ls'
        self.solver_params = {'eta': 0.05, 'eta_min': 0.05, 'eps': 1e-9}

        self._poisoning_params = {
            "classifier": self.classifier,
            "training_data": self.tr,
            "surrogate_classifier": self.classifier,
            "surrogate_data": self.tr,
            "val": self.ts,
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

    def _set_up(self, poisoning_class, clf_idx, clf_class, clf_params):

        self.plot_creation = False

        self.clf_idx = clf_idx
        self.pois_class = poisoning_class
        self.clf_class = clf_class
        self.clf_params = clf_params

    def _test_init(self, normalizer=None):
        """Creates the classifier and fit it. """

        self._dataset_creation()

        # create the classifier
        self.classifier = self.clf_class(preprocess=normalizer,
                                         **self.clf_params)
        self.classifier.store_dual_vars = True
        # fit the classifier
        self.classifier.fit(self.tr)

    def _clf_poisoning(self):
        """
        Computes a poisoning point considering as source the sample {xc, yc}.
        """
        xc = self.poisoning._run(self.xc, self.yc)

        self.logger.info("Starting score: " + str(self.poisoning.f_seq[0]))
        self.logger.info("Final score: " + str(self.poisoning.f_seq[-1]))
        self.logger.info("x*: " + str(xc))
        self.logger.info("Point sequence: " + str(self.poisoning.x_seq))
        self.logger.info("Score sequence: : " + str(self.poisoning.f_seq))
        self.logger.info("Fun Eval: " + str(self.poisoning.f_eval))
        self.logger.info("Grad Eval: " + str(self.poisoning.grad_eval))

        metric = CMetric.create('accuracy')
        y_pred, scores = self.classifier.predict(self.ts.X,
                                                 return_decision_function=True)
        orig_acc = metric.performance_score(y_true=self.ts.Y,
                                            y_pred=y_pred)
        self.logger.info("Error on testing data: " + str(1 - orig_acc))

        tr = self.tr.append(CDataset(xc, self.yc))

        pois_clf = self.classifier.deepcopy()

        pois_clf.fit(tr)
        y_pred, scores = pois_clf.predict(self.ts.X,
                                          return_decision_function=True)
        pois_acc = metric.performance_score(y_true=self.ts.Y,
                                            y_pred=y_pred)
        self.logger.info(
            "Error on testing data (poisoned): " + str(1 - pois_acc))

        return pois_clf, xc

    def _test_attack_effectiveness(self, normalizer):
        """
        This function, firsly, computes a poisoning point. Than, it compares
        the value of the attacker objective function on that point before and
        after the attack. Finally, raises an error if the one computed on
        the poisoning point is not the highest.
        """
        self.logger.info("Test if the value of the attacker objective "
                         "function increases after the attack")

        self._test_init(normalizer)
        self._create_poisoning_object()

        x0 = self.xc  # starting poisoning point
        xc = self._clf_poisoning()[1]

        fobj_x0 = self.poisoning._objective_function(xc=x0)
        fobj_xc = self.poisoning._objective_function(xc=xc)

        self.logger.info(
            "Objective function before the attack {:}".format(fobj_x0))
        self.logger.info(
            "Objective function after the attack {:}".format(fobj_xc))

        self.assertLess(fobj_x0, fobj_xc,
                        "The attack does not increase the objective "
                        "function of the attacker. The fobj on the "
                        "original poisoning point is {:} while "
                        "on the optimized poisoning point is {:}.".format(
                            fobj_x0, fobj_xc))

        if self.plot_creation:
            self._create_2D_plots(normalizer)

    def _test_clf_accuracy(self, normalizer):
        """Checks the accuracy of the classifier considered into the
        test. """

        self._test_init(normalizer)

        metric = CMetric.create('accuracy')
        y_pred, scores = self.classifier.predict(self.ts.X,
                                                 return_decision_function=True)
        acc = metric.performance_score(y_true=self.ts.Y, y_pred=y_pred)
        self.logger.info("Error on testing data: " + str(1 - acc))
        self.assertGreater(
            acc, 0.70, "The trained classifier have an accuracy that "
                       "is too low to evaluate if the poisoning against "
                       "this classifier works")

    #####################################################################
    #                        PLOT FUNCTIONALITIES
    #####################################################################

    def _create_box(self):
        """Create a box constraint"""
        if self.lb == "x0":
            self.lb = self.x0
        if self.ub == "x0":
            self.ub = self.x0
        box = CConstraintBox(lb=self.lb, ub=self.ub)
        return box

    def _plot_func(self, fig, func, **func_kwargs):
        """Plot poisoning objective function"""
        fig.sp.plot_fun(
            func=func,
            grid_limits=self.grid_limits, plot_levels=False,
            n_grid_points=10, colorbar=True, **func_kwargs)

    def _plot_obj_grads(self, fig, func, **func_kwargs):
        """Plot poisoning attacker objective function gradient"""
        fig.sp.plot_fgrads(
            func,
            grid_limits=self.grid_limits,
            n_grid_points=20, **func_kwargs)

    def _create_2D_plots(self, normalizer):

        self._test_init(normalizer)

        self.logger.info("Create 2-dimensional plot")

        self.clf_orig = self.classifier.deepcopy()
        pois_clf = self._clf_poisoning()[0]

        fig = CFigure(height=4, width=10, title=self.clf_idx)
        n_rows = 1
        n_cols = 2

        box = self._create_box()

        fig.subplot(n_rows, n_cols, grid_slot=1)
        fig.sp.title('Attacker objective and gradients')
        self._plot_func(fig, self.poisoning._objective_function)
        self._plot_obj_grads(
            fig, self.poisoning._objective_function_gradient)
        fig.sp.plot_ds(self.tr)
        fig.sp.plot_decision_regions(self.clf_orig, plot_background=False,
                                     grid_limits=self.grid_limits,
                                     n_grid_points=10, )

        fig.sp.plot_constraint(box, grid_limits=self.grid_limits,
                               n_grid_points=10)
        fig.sp.plot_path(self.poisoning.x_seq,
                         start_facecolor='r' if self.yc == 1 else 'b')

        fig.subplot(n_rows, n_cols, grid_slot=2)
        fig.sp.title('Classification error on val')
        self._plot_func(fig, self.poisoning._objective_function,
                        acc=True)
        fig.sp.plot_ds(self.tr)
        fig.sp.plot_decision_regions(pois_clf, plot_background=False,
                                     grid_limits=self.grid_limits,
                                     n_grid_points=10, )

        fig.sp.plot_constraint(box, grid_limits=self.grid_limits,
                               n_grid_points=10)
        fig.sp.plot_path(self.poisoning.x_seq,
                         start_facecolor='r' if self.yc == 1 else 'b')

        fig.tight_layout()
        exp_idx = "2d_pois_"
        exp_idx += self.clf_idx
        if self.classifier.preprocess is not None:
            exp_idx += "_norm"
        fig.savefig(exp_idx + '.pdf', file_format='pdf')

    #####################################################################
    # FUNCTIONS TO CHECK THE POISONING GRADIENT OF CLASSIFIERS
    # LEARNED IN THE PRIMAL
    #####################################################################

    def _plot_param_sub(self, fig, param_fun, grad_fun, clf):

        box = self._create_box()

        self._plot_func(fig, param_fun)
        self._plot_obj_grads(
            fig, grad_fun)

        fig.sp.plot_ds(self.tr)
        fig.sp.plot_decision_regions(clf, plot_background=False,
                                     grid_limits=self.grid_limits,
                                     n_grid_points=10, )
        fig.sp.plot_constraint(box, grid_limits=self.grid_limits,
                               n_grid_points=10)

    def _create_params_grad_plot(self, normalizer):
        """
        Show the gradient of the classifier parameters w.r.t the poisoning
        point
        """
        self.logger.info("Create 2-dimensional plot of the poisoning "
                         "gradient")

        self._test_init(normalizer)

        pois_clf = self._clf_poisoning()[0]

        if self.n_features == 2:
            debug_pois_obj = _CAttackPoisoningLinTest(self.poisoning)

            fig = CFigure(height=8, width=10)
            n_rows = 2
            n_cols = 2

            fig.title(self.clf_idx)

            fig.subplot(n_rows, n_cols, grid_slot=1)
            fig.sp.title('w1 wrt xc')
            self._plot_param_sub(fig, debug_pois_obj.w1,
                                 debug_pois_obj.gradient_w1_xc,
                                 pois_clf)

            fig.subplot(n_rows, n_cols, grid_slot=2)
            fig.sp.title('w2 wrt xc')
            self._plot_param_sub(fig, debug_pois_obj.w2,
                                 debug_pois_obj.gradient_w2_xc,
                                 pois_clf)

            fig.subplot(n_rows, n_cols, grid_slot=3)
            fig.sp.title('b wrt xc')
            self._plot_param_sub(fig, debug_pois_obj.b,
                                 debug_pois_obj.gradient_b_xc,
                                 pois_clf)

            fig.tight_layout()
            exp_idx = "2d_grad_pois_"
            exp_idx += self.clf_idx
            if self.classifier.preprocess is not None:
                exp_idx += "_norm"
            fig.savefig(exp_idx + '.pdf', file_format='pdf')

    def _single_param_grad_check(self, xc, f_param, df_param, param_name):
        """

        Parameters
        ----------
        xc CArray
            poisoning point
        f_param function
            the function that update the parameter value
        df_param function
            the function that compute the gradient value
        param_name the parameter name
        """

        # Compare analytical gradient with its numerical approximation
        check_grad_val = CFunction(
            f_param, df_param).check_grad(xc, epsilon=100)
        self.logger.info("Gradient difference between analytical {:} "
                         "gradient and numerical gradient: %s".format(
            param_name),
            str(check_grad_val))
        self.assertLess(check_grad_val, 1,
                        "poisoning gradient is wrong {:}".format(
                            check_grad_val))

    def _test_single_poisoning_grad_check(self, normalizer):

        self._test_init(normalizer)

        pois_clf = self._clf_poisoning()[0]

        xc = self.xc

        debug_pois_obj = _CAttackPoisoningLinTest(self.poisoning)

        self._single_param_grad_check(xc, debug_pois_obj.w1,
                                      debug_pois_obj.gradient_w1_xc,
                                      param_name='w1')
        self._single_param_grad_check(xc, debug_pois_obj.w2,
                                      debug_pois_obj.gradient_w2_xc,
                                      param_name='w2')
        self._single_param_grad_check(xc, debug_pois_obj.b,
                                      debug_pois_obj.gradient_b_xc,
                                      param_name='b')

        if self.plot_creation is True:
            self._create_params_grad_plot(normalizer)


if __name__ == '__main__':
    CUnitTest.main()
