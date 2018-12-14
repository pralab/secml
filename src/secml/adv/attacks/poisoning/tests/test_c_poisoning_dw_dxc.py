from secml.utils import CUnitTest
from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CDataSplitterShuffle
from secml.ml.features.normalization import CNormalizerMinMax
from test_c_poisoning import CPoisoningTestCases
from secml.adv.attacks.poisoning.tests import CAttackPoisoningLinTest
from secml.figure import CFigure
from secml.optimization import COptimizer
from secml.optimization.function import CFunction

class TestCPoisoning_dw_dxc(CPoisoningTestCases.TestCPoisoning):
    """
    Check the derivative of the classifier weights w.r.t. the poisoning point

    (d_w w.r.t d_xc and d_b w.r.t d_xc)
    """

    def param_setter(self):
        self.clf_idx = 'ridge'  # logistic | ridge | svm

    def _dataset_creation(self):
        self.n_features = 2  # Number of dataset features

        self.n_tr = 100
        self.n_ts = 100
        self.n_classes = 2

        # Random state generator for the dataset
        self.seed = 44

        if self.n_classes == 2:
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
        self.tr.X = normalizer.train_normalize(self.tr.X)
        self.ts.X = normalizer.normalize(self.ts.X)

        self.lb = -1
        self.ub = 1

        self.grid_limits = [(self.lb - 0.1, self.ub + 0.1),
                            (self.lb - 0.1, self.ub + 0.1)]

    def test_poisoning_2D_plot(self):

        pois_clf = self._clf_poisoning()

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
            fig.savefig(self.name_file, file_format='pdf')

    def _single_param_grad_check(self, xc, f_param, df_param, param_name):

        # Compare analytical gradient with its numerical approximation
        check_grad_val = COptimizer(
            CFunction(f_param,
                      df_param)
        ).check_grad(xc)
        self.logger.info("Gradient difference between analytical {:} "
                         "gradient and numerical gradient: %s".format(
            param_name),
                         str(check_grad_val))
        self.assertLess(check_grad_val, 1e-3,
                        "poisoning gradient is wrong {:}".format(
                            check_grad_val))
        for i, elm in enumerate(self.xc.size):
            self.assertIsInstance(elm, float)

    def test_poisoning_grad_check(self):

        pois_clf = self._clf_poisoning()

        xc = self.xc

        debug_pois_obj = CAttackPoisoningLinTest(self.poisoning)

        self._single_param_grad_check(xc, debug_pois_obj.w1,
                                      debug_pois_obj.gradient_w1_xc,
                                      param_name='w1')
        self._single_param_grad_check(xc, debug_pois_obj.w2,
                                      debug_pois_obj.gradient_w2_xc,
                                      param_name='w2')
        self._single_param_grad_check(xc, debug_pois_obj.b,
                                  debug_pois_obj.gradient_b_xc, param_name='b')


if __name__ == '__main__':
    CUnitTest.main()