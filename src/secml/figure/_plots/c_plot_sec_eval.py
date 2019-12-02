"""
.. module:: CPlotSecEval
   :synopsis: Classifier Security Evaluation plots.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.figure._plots import CPlot
from secml.ml.peval.metrics import CMetric
from secml.array import CArray


def _cmpt_sec_eval_curve(sec_eval_data, metric, label=1):
    """Compute metric for each security evaluation parameter.

    Parameters
    ----------
    sec_eval_data : CSecEvalData
        Security Evaluation data object.
    metric : CMetric
        Metric object.
    label : int, optional
        Label wrt the metric should be computed. Default 1.

    """
    perf = CArray.zeros(shape=(sec_eval_data.param_values.size,))
    for k in range(sec_eval_data.param_values.size):

        scores = sec_eval_data.scores[k][:, label].ravel()
        y_pred = sec_eval_data.Y_pred[k].ravel()

        metric_val = metric.performance_score(
            y_true=sec_eval_data.Y, y_pred=y_pred, score=scores)

        perf[k] = metric_val

    return perf


class CPlotSecEval(CPlot):
    """Plots Classifier Security Evaluation results.

    This class creates a figure plotting in a fancy and standard
    style data from `.CSecEvalData` class.

    Custom plotting parameters can be specified.

    Currently parameters default:
     - `show_legend`: True. Set False to hide `show_legend` on next plot.
     - grid: True.

    See Also
    --------
    .CPlot : basic subplot functions.
    .CFigure : creates and handle figures.

    """

    def apply_params_sec_eval(self):
        """Apply defined parameters to active subplot."""
        fig_legend = self.get_legend()
        if self.show_legend is not False and fig_legend is not None:
            fig_legend.set_visible(True)
        self.grid(grid_on=True)

    def plot_sec_eval(self, sec_eval_data, metric='accuracy', mean=False,
                      percentage=False, show_average=False, label=None,
                      linestyle='-', color=None, marker=None, metric_args=()):
        """Plot the Security Evaluation Curve using desired metric.

        Parameters
        ----------
        sec_eval_data : CSecEvalData or list
            A single CSecEvalData object or a list with multiple repetitions.
        metric : str or CMetric, optional
            Metric to be evaluated. Default 'accuracy'.
        mean : bool, optional
            If True, the mean of all sec eval repetitions will be computed.
            Default False..
        percentage : bool, optional
            If True, values will be displayed in percentage. Default False.
        show_average : bool, optional
            If True, the average along the sec eval parameters will be
            shown in legend. Default False.
        label : str, optional
            Label of the sec eval curve. Default None.
        linestyle : str, optional
            Style of the curve. Default '-'.
        color : str or None, optional
            Color of the curve. If None (default) the plot engine will chose.
        marker : str or None, optional
            Style of the markers. Default None.
        metric_args
            Any other argument for the metric.

        """
        metric = CMetric.create(metric, *metric_args)

        if not isinstance(sec_eval_data, list):
            sec_eval_data = [sec_eval_data]

        n_sec_eval = len(sec_eval_data)
        n_param_val = sec_eval_data[0].param_values.size
        perf = CArray.zeros((n_sec_eval, n_param_val))

        for i in range(n_sec_eval):
            if sec_eval_data[i].param_values.size != n_param_val:
                raise ValueError("the number of sec eval parameters changed!")

            perf[i, :] = _cmpt_sec_eval_curve(sec_eval_data[i], metric)

        if mean is True:
            perf_std = perf.std(axis=0, keepdims=False)
            perf = perf.mean(axis=0, keepdims=False)
        else:
            if len(sec_eval_data) > 1:
                raise ValueError("if `mean` is False, "
                                 "only one sec eval data should be passed")

        perf = perf.ravel()

        if percentage is True:
            perf *= 100
            if mean is True:
                perf_std *= 100

        if show_average is True:
            auc_val = perf.mean()
            if label is None:
                label = "err: {:.2f}".format(auc_val)
            else:
                label += ", err: {:.2f}".format(auc_val)

        # This is done here to make 'markevery' work correctly
        self.xticks(sec_eval_data[0].param_values)

        self.plot(sec_eval_data[0].param_values, perf, label=label,
                  linestyle=linestyle, color=color, marker=marker,
                  markevery=self.get_xticks_idx(sec_eval_data[0].param_values))

        if mean is True:
            std_up = perf + perf_std
            std_down = perf - perf_std
            std_down[std_down < 0.0] = 0.0
            if percentage is True:
                std_up[std_up > 100] = 100
            else:
                std_up[std_up > 1.0] = 1.0
            self.fill_between(sec_eval_data[0].param_values, std_up, std_down,
                              interpolate=False, alpha=0.2, facecolor=color,
                              linestyle='None')

        if self._xlabel is None:
            self.xlabel(sec_eval_data[0].param_name)
        if self._ylabel is None:
            self.ylabel(metric.class_type.capitalize())

        self.legend(loc='best', labelspacing=0.4,
                    handletextpad=0.3, edgecolor='k')
        self.title("Security Evaluation Curve")

        self.apply_params_sec_eval()
