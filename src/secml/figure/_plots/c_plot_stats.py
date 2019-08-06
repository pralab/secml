"""
.. module:: CPlotStats
   :synopsis: Statistical functions plots.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.figure._plots import CPlot
from secml.ml.stats import CDensityEstimation


class CPlotStats(CPlot):
    """Plots for statistical functions.

    Custom plotting parameters can be specified.

    Currently parameters default:
     - `show_legend`: True.
     - grid: True.

    See Also
    --------
    CPlot : basic subplot functions.
    CFigure : creates and handle figures.

    """

    def apply_params_stats(self):
        """Apply defined parameters to active subplot."""
        fig_legend = self.get_legend()
        if self.show_legend is not False and fig_legend is not None:
            fig_legend.set_visible(True)
        self.grid(grid_on=True)

    def plot_prob_density(self, scores, ts, **params):
        """Plot density estimation of benign and malicious class."""
        de = CDensityEstimation(**params)
        xm, malicious_pdf = de.estimate_density(scores[ts.Y == 1])
        xb, benign_pdf = de.estimate_density(scores[ts.Y == 0])

        self.plot(xb, benign_pdf, label="ben pdf")
        self.plot(xm, malicious_pdf, label="mal pdf")

        # Customizing figure
        self.apply_params_stats()
