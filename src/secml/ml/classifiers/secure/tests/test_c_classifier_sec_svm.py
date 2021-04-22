from secml.testing import CUnitTest

from secml.data.loader import CDLRandom
from secml.figure import CFigure
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.secure import CClassifierSecSVM


class TestCClassifierSecSVM(CUnitTest):
    """Unittests for CClassifierSecSVM."""

    def setUp(self):
        pass

    def _compute_alignment(self, ds, secsvm, svm):

        self.logger.info(
            "Sec-SVM, Avg. Hinge loss: \n{:}".format(secsvm.hinge_loss(
                ds.X, 2 * ds.Y - 1).mean()))

        self.logger.info("SVM, b: {:}".format(svm.b))
        self.logger.info("SVM, w: \n{:}".format(svm.w))
        self.logger.info("Sec-SVM, b: {:}".format(secsvm.b))
        self.logger.info("Sec-SVM, w: \n{:}".format(secsvm.w))

        angle = secsvm.w.dot(svm.w.T) / (secsvm.w.norm() * svm.w.norm())
        self.logger.info("Angle between hyperplanes: {:}".format(angle))

        self.assertGreater(angle, 0.7)

        self.logger.info(
            "Objective Function: \n{:}".format(secsvm.objective(ds.X, ds.Y)))
        self.logger.info(
            "Gradient w vs b: \n{:}".format(secsvm.gradient_w_b(ds.X, ds.Y)))

    def test_alignment(self):

        ds = CDLRandom(n_samples=100,
                       n_features=500,
                       n_redundant=0,
                       n_informative=10,
                       n_clusters_per_class=1,
                       random_state=0).load()

        self.logger.info("Train Sec SVM")
        sec_svm = CClassifierSecSVM(C=1, eta=0.1, eps=1e-2, lb=-0.1, ub=0.5)
        sec_svm.verbose = 2
        sec_svm.fit(ds.X, ds.Y)

        self.logger.info("Train SVM")
        svm = CClassifierSVM(C=1)
        svm.fit(ds.X, ds.Y)

        self._compute_alignment(ds, sec_svm, svm)

        svm_pred = sec_svm.predict(ds.X)
        secsvm_pred = sec_svm.predict(ds.X)

        self.logger.info("SVM pred:\n{:}".format(svm_pred))
        self.logger.info("Sec-SVM pred:\n{:}".format(secsvm_pred))

        self.assert_array_almost_equal(secsvm_pred, svm_pred)

    def test_plot(self):

        ds = CDLRandom(n_samples=100,
                       n_features=2,
                       n_redundant=0,
                       random_state=100).load()

        self.logger.info("Train Sec SVM")
        sec_svm = CClassifierSecSVM(C=1, eta=0.1, eps=1e-3, lb=-0.1, ub=0.5)
        sec_svm.verbose = 2
        sec_svm.fit(ds.X, ds.Y)

        self.logger.info("Train SVM")
        svm = CClassifierSVM(C=1)
        svm.fit(ds.X, ds.Y)

        self._compute_alignment(ds, sec_svm, svm)

        fig = CFigure(height=5, width=8)
        fig.subplot(1, 2, 1)
        # Plot dataset points
        fig.sp.plot_ds(ds)
        # Plot objective function
        fig.sp.plot_fun(svm.predict,
                        multipoint=True,
                        plot_background=True,
                        plot_levels=False,
                        n_grid_points=100,
                        grid_limits=ds.get_bounds())
        fig.sp.title("SVM")

        fig.subplot(1, 2, 2)
        # Plot dataset points
        fig.sp.plot_ds(ds)
        # Plot objective function
        fig.sp.plot_fun(sec_svm.predict,
                        multipoint=True,
                        plot_background=True,
                        plot_levels=False,
                        n_grid_points=100,
                        grid_limits=ds.get_bounds())
        fig.sp.title("Sec-SVM")

        fig.show()


if __name__ == '__main__':
    CUnitTest.main()
