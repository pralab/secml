import unittest
from secml.utils import CUnitTest

from secml.figure.c_figure import CFigure
from secml.data.selection import CPrototypesSelector
from secml.data.loader import CDLRandomBlobs


class TestPS(CUnitTest):

    def setUp(self):
        self.dataset = CDLRandomBlobs(
            n_features=2, centers=[[-1, 1], [1, 1]],
            cluster_std=(0.4, 0.4)).load()

    def test_ps_random(self):
        self.logger.info("Testing Random Selection")

        rule = 'random'

        ps = CPrototypesSelector.create(rule)
        ps.verbose = 2
        ds_reduced = ps.select(
            self.dataset, n_prototypes=20, random_state=200)

        self.draw_selection(ds_reduced, rule)

    def test_ps_border(self):
        self.logger.info("Testing Border Selection")

        rule = 'border'

        ps = CPrototypesSelector.create(rule)
        ps.verbose = 2
        ds_reduced = ps.select(self.dataset, n_prototypes=20)

        self.draw_selection(ds_reduced, rule)

    def test_ps_center(self):
        self.logger.info("Testing Center Selection")

        rule = 'center'

        ps = CPrototypesSelector.create(rule)
        ps.verbose = 2
        ds_reduced = ps.select(self.dataset, n_prototypes=20)

        self.draw_selection(ds_reduced, rule)

    def test_ps_spanning(self):
        self.logger.info("Testing Spanning Selection")

        rule = 'spanning'

        ps = CPrototypesSelector.create(rule)
        ps.verbose = 2
        ds_reduced = ps.select(self.dataset, n_prototypes=20)

        self.draw_selection(ds_reduced, rule)

    def test_ps_kmedians(self):
        self.logger.info("Testing K-Median Selection")

        rule = 'k-medians'

        ps = CPrototypesSelector.create(rule)
        ps.verbose = 2
        ds_reduced = ps.select(self.dataset, n_prototypes=20)

        self.draw_selection(ds_reduced, rule)

    def draw_selection(self, ds_reduced, rule):
        fig = CFigure(width=10, markersize=12)
        fig.switch_sptype(sp_type='ds')
        # Plot dataset points
        fig.sp.plot_ds(self.dataset)
        fig.sp.plot(ds_reduced.X[:, 0], ds_reduced.X[:, 1], linestyle='None',
                    markeredgewidth=2, marker='o', mfc='none')
        fig.sp.title('PS rule: {:}'.format(rule))

        fig.show()


if __name__ == '__main__':
    unittest.main()
