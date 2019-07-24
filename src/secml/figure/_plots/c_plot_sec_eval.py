"""
.. module:: CPlotSecEval
   :synopsis: Classifier Security Evaluation plots.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from six.moves import range

from secml.figure._plots import CPlot
from secml.ml.peval.metrics import CMetric
from secml.array import CArray


# FIXME: CLEANUP NEEDED
class CPlotSecEval(CPlot):
    """Plots Classifier Security Evaluation results.

    This class creates a figure plotting in a fancy and standard
    style data from `.CSecEvalData` class.

    Custom plotting parameters can be specified.
    Currently parameters default:
     - show_CPlotSecEval: True. Set False to hide CPlotSecEval on next plot.
     - grid: True.

    See Also
    --------
    .CRoc : computes the receiver operating characteristic curve, or ROC curve.
    .CPlot : basic subplot functions.
    .CFigure : creates and handle figures.

    """

    def apply_params_sec_eval(self):
        """Apply defined parameters to active subplot."""
        fig_legend = self.get_legend()
        if self.show_legend is not False and fig_legend is not None:
            fig_legend.set_visible(True)
        self.grid(grid_on=True)

    def plot_metric(self, sec_eval_data, consider_target=False,
                    metric="accuracy", label=None, auc=False,
                    plot_std=False, linestyle='-', color=None):
        """Plot the Security Evaluation curve computed on all the samples
        with the chosen Security Evaluation metric.

        Parameters
        ----------
        :param sec_eval_data: CSecEvalData or list of CSecEvalData.

        """
        metric = CMetric.create(metric)

        # create security evaluation plot
        if not isinstance(sec_eval_data, list):
            sec_eval_data = [sec_eval_data]

        if not self._xlabel:
            self._xlabel = sec_eval_data[0].param_name
        if not self._ylabel:
            self._ylabel = metric.class_type

        samples_idx = CArray.arange(sec_eval_data[0].Y.size)

        perf, perf_std = self._compute_sec_eval_curve(
            sec_eval_data, samples_idx, consider_target, metric)

        # FIXME: THIS IS NOT THE AUC
        if auc is True:
            auc_val = perf.mean()
            if label is None:
                label = "err = {:.2f}".format(auc_val)
            else:
                label += ", err = {:.2f}".format(auc_val)

        self.plot(sec_eval_data[0].param_values, perf, label=label,
                  linestyle=linestyle, color=color)

        if plot_std is True:
            self.fill_between(sec_eval_data[0].param_values,
                              perf + perf_std, perf - perf_std,
                              interpolate=False, alpha=0.2, facecolor=color,
                              linestyle='None')

        if label is not None:
            self.legend(loc=4, labelspacing=0.4, handletextpad=0.3)

        self.apply_params_sec_eval()

    def plot_metric_for_class(self, sec_eval_data,
                              consider_target=True,
                              metric_name="accuracy"):
        """
        Plot the Security Evaluation curve computed on all the samples with the chosen Security Evaluation metric.

        Parameters
        ----------
        :param sec_eval_data: CSecEvalData or list of CSecEvalData.

        """
        metric = CMetric.create(metric_name)

        if not isinstance(sec_eval_data, list):
            sec_eval_data = [sec_eval_data]

        # create security evaluation plot
        if not self._xlabel:
            self._xlabel = sec_eval_data[0].param_name
        if not self._ylabel:
            self._ylabel = metric.class_type

        clss = sec_eval_data[0].Y.unique()
        for cls in clss:
            samples_idx = sec_eval_data[0].Y.find(sec_eval_data[0].Y == cls)

            perf = self._compute_sec_eval_curve(sec_eval_data, samples_idx,
                                                consider_target, metric)
            self.plot(sec_eval_data[0].param_values, perf[0], label=str(cls))

        self.legend(loc=1, labelspacing=0.4, handletextpad=0.3)
        self.apply_params_sec_eval()

    def cmpt_au_sec(self, sec_eval_data, consider_target=False,
                    metric_name="test_error"):
        """
        Compute the are under the average sec eval curve

        Parameters
        ----------
        :param sec_eval_data: CSecEvalData or list of CSecEvalData.

        """
        metric = CMetric.create(metric_name)

        # create security evaluation plot
        if not isinstance(sec_eval_data, list):
            sec_eval_data = [sec_eval_data]

        if not self._xlabel:
            self._xlabel = sec_eval_data[0].param_name
        if not self._ylabel:
            self._ylabel = metric.class_type

        samples_idx = CArray.arange(sec_eval_data[0].Y.size)

        perf, perf_std = self._compute_sec_eval_curve(
            sec_eval_data, samples_idx, consider_target, metric)

        return perf.mean(axis=None)

    def _compute_sec_eval_curve(self, sec_eval_data, samples_idx, consider_target, metric):
        """
        Evaluates performance under attack evaluated with a chosen metric.

        Parameters
        ----------
        :param sec_eval_data: CSecEvalData or list of CSecEvalData.
        :param metric : CMetric

        """
        if not isinstance(sec_eval_data, list):
            sec_eval_data = [sec_eval_data]

        n_sec_eval_data = len(sec_eval_data)
        n_param_val = sec_eval_data[0].param_values.size

        perf = CArray.zeros((n_sec_eval_data, n_param_val))

        for sec_eval_idx, sngl_sec_eval_data in enumerate(sec_eval_data):
            perf[sec_eval_idx, :] = self._compute_one_sec_eval_curve(
                sngl_sec_eval_data, samples_idx, consider_target, metric)

        return perf.mean(axis=0, keepdims=False), \
            perf.std(axis=0, keepdims=False)

    def _compute_one_sec_eval_curve(self, sngl_sec_eval_data, samples_idx, consider_target, metric):
        """
        Evaluates performance under attack evaluated with a chosen metric for one repetition.

        :param sec_eval_data: CSecEvalData
        :param metric: CMetric
        """
        perf = CArray.zeros(shape=(sngl_sec_eval_data.param_values.size,))
        for k in range(sngl_sec_eval_data.param_values.size):
            s = sngl_sec_eval_data.scores[k]  # (num_samples, num_classes)
            # consider only sample with idx in samples_idx
            s = s[samples_idx, :].ravel()
            l = CArray(sngl_sec_eval_data.Y_pred[k].ravel())  # (num_samples,)
            l = l[samples_idx].ravel()

            if consider_target and sngl_sec_eval_data.Y_target is not None:
                y_target = sngl_sec_eval_data.Y_target[samples_idx].ravel()
            else:
                y_target = sngl_sec_eval_data.Y[samples_idx].ravel()

            # otherwise compute performance measure
            metric_val = metric.performance_score(
                y_true=y_target, y_pred=l, score=s)

            perf[k] = metric_val
        return perf
