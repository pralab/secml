from secml.utils import CUnitTest
from test_c_poisoning import CPoisoningTestCases

from secml.figure import CFigure
from secml.optimization import COptimizer
from secml.optimization.function import CFunction


class TestCPoisoningBlob(CPoisoningTestCases.TestCPoisoning):

    def clf_list(self):
        return ['ridge', 'logistic']

    def test_poisoning_2D_plot(self):
        self.logger.info("Create 2-dimensional plot")

        for clf_idx in self.clf_list():
            self.logger.info("Test the {:} classifier".format(clf_idx))
            self._objs_creation(clf_idx)

            pois_clf = self._clf_poisoning()[0]

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
                fig.savefig(clf_idx + "_2d_pois", file_format='pdf')

    def test_poisoning_point_fobj_improvement(self):
        """
        This function check if the objective function of the original
        classifier is higger when it is trained on the optimized
        poisoning point than when it is trained on the starting
        poisoning point.
        """
        self.logger.info("Test if the value of the attacker objective "
                         "function improves after the attack")

        for clf_idx in self.clf_list():
            self.logger.info("Test the {:} classifier".format(clf_idx))
            self._objs_creation(clf_idx)

            x0 = self.xc  # starting poisoning point
            xc = self._clf_poisoning()[1]

            fobj_x0 = self.poisoning._objective_function(xc=x0)
            fobj_xc = self.poisoning._objective_function(xc=xc)

            self.assertLess(fobj_x0, fobj_xc,
                            "The attack does not increase the objective "
                            "function of the attacker. The fobj on the "
                            "original poisoning point is {:} while "
                            "on the optimized poisoning point is {:}.".format(
                                fobj_x0, fobj_xc))

    def test_acc_impact(self):
        """
        Check if the accuracy of the classifier decrease when it is
        trained on the poisoning point.
        """
        self.logger.info("Test the impact of the attack on the classifier "
                         "accuracy")

        for clf_idx in self.clf_list():
            self.logger.info("Test the {:} classifier".format(clf_idx))
            self._objs_creation(clf_idx)

            x0 = self.xc  # starting poisoning point
            xc = self._clf_poisoning()[1]

            acc_tr_on_x0 = self.poisoning._objective_function(xc=x0, acc=True)
            acc_tr_on_xc = self.poisoning._objective_function(xc=xc, acc=True)

            self.assertLess(acc_tr_on_x0, acc_tr_on_xc,
                            "The attack does not decrease the classifier "
                            "accuracy. The accuracy of the classifier trained "
                            "on the original poisoning point is {:} while "
                            "on the optimized poisoning point is {:}.".format(
                                acc_tr_on_x0, acc_tr_on_xc))

    def test_poisoning_grad_check(self):

        self.logger.info("Compare the numerical with the analytical "
                         "poisoning gradient")

        for clf_idx in self.clf_list():
            self.logger.info("Test the {:} classifier".format(clf_idx))
            self._objs_creation(clf_idx)

            self._clf_poisoning()

            x0 = self.xc

            # Compare analytical gradient with its numerical approximation
            check_grad_val = COptimizer(
                CFunction(self.poisoning._objective_function,
                          self.poisoning._objective_function_gradient)
            ).check_grad(x0)
            self.logger.info("Gradient difference between analytical "
                             "poisoning "
                             "gradient and numerical gradient: %s",
                             str(check_grad_val))
            self.assertLess(check_grad_val, 1,
                            "poisoning gradient is wrong {:}".format(
                                check_grad_val))


if __name__ == '__main__':
    CUnitTest.main()
