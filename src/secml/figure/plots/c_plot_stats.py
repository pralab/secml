from secml.figure.plots import CPlot
from secml.ml.stats import CDensityEstimation


class CPlotStats(CPlot):
    """Plots for statistical functions.

    Custom plotting parameters can be specified.
    Currently parameters default:
     - show_legend: True. Set False to hide legend on next plot.
     - grid: True.

    Parameters
    ----------
    sp : Axes
        Subplot to use for plotting. Instance of `matplotlib.axes.Axes`.
    default_params : dict
        Dictionary with default parameters.

    Attributes
    ----------
    class_type : 'stats'

    See Also
    --------
    .CPlot : basic subplot functions.
    .CFigure : creates and handle figures.

    """
    __class_type = 'stats'

    def __init__(self, sp, default_params=None):

        # Calling CPlot constructor
        super(CPlotStats, self).__init__(
            sp=sp, default_params=default_params)

        # Specific plot parameters (use `set_params` to alter)
        self.show_legend = True
        self.grid(grid_on=True)

    def _apply_params(self):
        """Apply defined parameters to active subplot."""
        fig_legend = self.get_legend()
        if fig_legend is not None:
            fig_legend.set_visible(self.show_legend)

    def plot_prob_density(self, scores, ts, **params):
        """Plot density estimation of benign and malicious class."""
        de = CDensityEstimation(**params)
        xm, malicious_pdf = de.estimate_density(scores[ts.Y == 1])
        xb, benign_pdf = de.estimate_density(scores[ts.Y == 0])

        self.plot(xb, benign_pdf, label="ben pdf")
        self.plot(xm, malicious_pdf, label="mal pdf")

        # Customizing figure
        self._apply_params()
