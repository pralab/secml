from secml.utils import CUnitTest
from test_c_poisoning import CPoisoningTestCases

from secml.figure import CFigure


class TestCPoisoningBlob(CPoisoningTestCases.TestCPoisoning):

    @property
    def clf_list(self):
        return ['ridge', 'logistic', 'lin-svm', 'rbf-svm']

    def test_poisoning_2D_plot(self):
        self.plot = False
        if self.plot:
            self._make_plot()

    def _make_plot(self):
        self.logger.info("Create 2-dimensional plot")

        normalizer_vals = [False, True]
        combinations_list = [(clf_idx, normalizer) for clf_idx in \
                             self.clf_list for normalizer in normalizer_vals]

        for clf_idx, normalizer in combinations_list:
            if normalizer:
                self.logger.info("Test the {:} classifier when it has "
                                 "a normalizer inside ".format(clf_idx))
            else:
                self.logger.info("Test the {:} classifier when it does "
                                 "not have a normalizer inside ".format(clf_idx))
            self._objs_creation(clf_idx, normalizer)

            pois_clf = self._clf_poisoning()[0]

            if self.n_features == 2:
                fig = CFigure(height=4, width=10, title=clf_idx)
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

                fig.tight_layout()
                fig.show()
                exp_idx = "2d_pois_"
                exp_idx += clf_idx
                if normalizer:
                    exp_idx += "_norm"
                fig.savefig(exp_idx + '.pdf', file_format='pdf')

    def test_poisoning_point_fobj_improvement(self):
        """
        This function check if the objective function of the original
        classifier is higger when it is trained on the optimized
        poisoning point than when it is trained on the starting
        poisoning point.
        """
        self.logger.info("Test if the value of the attacker objective "
                         "function improves after the attack")

        normalizer_vals = [False, True]
        combinations_list = [(clf_idx, normalizer) for clf_idx in \
                             self.clf_list for normalizer in normalizer_vals]

        for clf_idx, normalizer in combinations_list:
            if normalizer:
                self.logger.info("Test the {:} classifier when it has "
                                 "a normalizer inside ".format(clf_idx))
            else:
                if normalizer:
                    self.logger.info("Test the {:} classifier when it does "
                                     "not have a normalizer inside ".format(clf_idx))
            self._objs_creation(clf_idx, normalizer)

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


if __name__ == '__main__':
    CUnitTest.main()
