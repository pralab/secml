from secml.testing import CUnitTest

from secml.array import CArray
from secml.figure.c_figure import CFigure
from secml.data.selection import CPrototypesSelector
from secml.data.loader import CDLRandomBlobs
from secml.utils import fm


class TestPS(CUnitTest):

    @classmethod
    def setUpClass(cls):
        cls.plots = False
        cls.dataset = CDLRandomBlobs(
            n_features=2, centers=[[-1, 1], [1, 1]],
            cluster_std=(0.4, 0.4), random_state=0).load()
        CUnitTest.setUpClass()  # call superclass constructor

    def _test_rule(self, rule, n_prototypes=20, random_state=None):
        """Generic test case for prototype selectors."""
        self.logger.info("Testing: " + rule + " selector.")
        ps = CPrototypesSelector.create(rule)
        ps.verbose = 2
        if random_state is None:
            ds_reduced = ps.select(self.dataset, n_prototypes=n_prototypes)
        else:
            ds_reduced = ps.select(self.dataset, n_prototypes=n_prototypes,
                                   random_state=random_state)
        if self.plots is True:
            self.draw_selection(ds_reduced, rule)

        idx_path = fm.join(fm.abspath(__file__), "idx_{:}.gz".format(rule))
        self.assert_array_equal(
            ps.sel_idx, CArray.load(idx_path, dtype=int).ravel())

    def test_ps_random(self):
        self._test_rule('random', random_state=200)

    def test_ps_border(self):
        self._test_rule('border')

    def test_ps_center(self):
        self._test_rule('center')

    def test_ps_spanning(self):
        self._test_rule('spanning')

    # TODO: refactor this test when reqs will ask for sklearn >= 0.22
    def test_ps_kmedians(self):
        rule = 'k-medians'
        self.logger.info("Testing: " + rule + " selector.")
        ps = CPrototypesSelector.create(rule)
        ps.verbose = 2
        ds_reduced = ps.select(self.dataset, n_prototypes=20, random_state=0)

        if self.plots is True:
            self.draw_selection(ds_reduced, rule)

        # k_means in sklearn > 0.24 returns a different result
        import sklearn
        from pkg_resources import parse_version
        if parse_version(sklearn.__version__) < parse_version("0.24"):
            idx_path = fm.join(
                fm.abspath(__file__), "idx_{:}.gz".format(rule))
        else:
            idx_path = fm.join(
                fm.abspath(__file__), "idx_{:}_sk0-24.gz".format(rule))

        self.assert_array_equal(
            ps.sel_idx, CArray.load(idx_path, dtype=int).ravel())

    def draw_selection(self, ds_reduced, rule):
        fig = CFigure(width=10, markersize=12)
        # Plot dataset points
        fig.sp.plot_ds(self.dataset, colors=['c', 'g'])
        fig.sp.plot(ds_reduced.X[:, 0], ds_reduced.X[:, 1],
                    linestyle='None', mfc='none',
                    markeredgewidth=2, markeredgecolor='k', marker='o')
        fig.sp.title('PS rule: {:}'.format(rule))
        fig.show()


if __name__ == '__main__':
    CUnitTest.main()
