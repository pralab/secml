from secml.utils import CUnitTest
from test_c_poisoning import CPoisoningTestCases
from secml.adv.attacks.poisoning.tests import CAttackPoisoningLinTest
from secml.figure import CFigure
from secml.optimization import COptimizer
from secml.optimization.function import CFunction


class TestCPoisoning_dw_dxc(CPoisoningTestCases.TestCPoisoning):
    """
    Check the derivative of the classifier weights w.r.t. the poisoning point

    NB: does not works for classifier learned in the dual space.

    (d_w w.r.t d_xc and d_b w.r.t d_xc)
    """

    def clf_list(self):
        return ['logistic', 'ridge']

    def test_gradient_2D_plot(self):
        if self.plot:
            self._make_plot()

    def _make_plot(self):
        """
        Test the poisoning derivative showing some 2-dimensiona plots
        """
        self.logger.info("Create 2-dimensional plot")

        for clf_idx in self.clf_list():
            self.logger.info("Test the {:} classifier".format(clf_idx))
            self._objs_creation(clf_idx)

            pois_clf = self._clf_poisoning()[0]

            if self.n_features == 2:
                debug_pois_obj = CAttackPoisoningLinTest(self.poisoning)

                fig = CFigure(height=8, width=10)
                n_rows = 2
                n_cols = 2

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

                fig.show()
                fig.savefig(clf_idx + "_2d_grad_pois", file_format='pdf')

    def _single_param_grad_check(self, xc, f_param, df_param, param_name):

        # Compare analytical gradient with its numerical approximation
        check_grad_val = COptimizer(
            CFunction(f_param,
                      df_param)
        ).check_grad(xc, epsilon=10)
        self.logger.info("Gradient difference between analytical {:} "
                         "gradient and numerical gradient: %s".format(
            param_name),
            str(check_grad_val))
        self.assertLess(check_grad_val, 1,
                        "poisoning gradient is wrong {:}".format(
                            check_grad_val))

    def test_poisoning_grad_check(self):

        self.logger.info("Create 2-dimensional plot")

        for clf_idx in self.clf_list():
            self.logger.info("Test the {:} classifier".format(clf_idx))
            self._objs_creation(clf_idx)

            pois_clf = self._clf_poisoning()[0]

            xc = self.xc

            debug_pois_obj = CAttackPoisoningLinTest(self.poisoning)

            self._single_param_grad_check(xc, debug_pois_obj.w1,
                                          debug_pois_obj.gradient_w1_xc,
                                          param_name='w1')
            self._single_param_grad_check(xc, debug_pois_obj.w2,
                                          debug_pois_obj.gradient_w2_xc,
                                          param_name='w2')
            self._single_param_grad_check(xc, debug_pois_obj.b,
                                          debug_pois_obj.gradient_b_xc,
                                          param_name='b')


if __name__ == '__main__':
    CUnitTest.main()
