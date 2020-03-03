import unittest
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
        idx_path = fm.join(fm.abspath(__file__), "idx_{:}.gz".format(rule))
        self.assert_array_equal(
            ps.sel_idx, CArray.load(idx_path, dtype=int).ravel())
        if self.plots is True:
            self.draw_selection(ds_reduced, rule)

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

        # this test will fail with sklearn < 0.22, because of an issue in
        # random_state setting inside the k-means algorithm
        import sklearn
        from pkg_resources import parse_version
        if not parse_version(sklearn.__version__) < parse_version("0.22"):
            idx_path = fm.join(fm.abspath(__file__), "idx_{:}.gz".format(rule))
            self.assert_array_equal(ps.sel_idx,
                                    CArray.load(idx_path, dtype=int).ravel())
        if self.plots is True:
            self.draw_selection(ds_reduced, rule)

    def draw_selection(self, ds_reduced, rule):
        fig = CFigure(width=10, markersize=12)
        # Plot dataset points
        fig.sp.plot_ds(self.dataset)
        fig.sp.plot(ds_reduced.X[:, 0], ds_reduced.X[:, 1], linestyle='None',
                    markeredgewidth=2, marker='o', mfc='none')
        fig.sp.title('PS rule: {:}'.format(rule))
        fig.show()


if __name__ == '__main__':
    unittest.main()
