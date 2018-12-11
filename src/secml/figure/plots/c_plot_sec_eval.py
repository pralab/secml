from secml.figure.plots import CPlot
from secml.ml.peval.metrics import CMetric
from secml.array import CArray

#fixme: da pulire
class CPlotSecEval(CPlot):
    """Plots Classifier Security Evaluation results.

    This class creates a figure plotting in a fancy and standard
    style data from `.CSecEvalData` class.

    Custom plotting parameters can be specified.

    #fixme: update
    Currently parameters default:
     - show_CPlotSecEval: True. Set False to hide CPlotSecEval on next plot.
     - grid: True.

    Parameters
    ----------
    sp : Axes
        Subplot to use for plotting. Instance of `matplotlib.axes.Axes`.
    default_params : dict
        Dictionary with default parameters.

    Attributes
    ----------
    class_type : 'sec-eval'

    See Also
    --------
    .CRoc : computes the receiver operating characteristic curve, or ROC curve.
    .CPlot : basic subplot functions.
    .CFigure : creates and handle figures.

    """
    __class_type = 'sec-eval'

    # fixme: gestire i params di def ecc
    def __init__(self, sp, default_params=None):

        # Calling CPlot constructor
        super(CPlotSecEval, self).__init__(
            sp=sp, default_params=default_params)

        # Specific plot parameters (use `set_params` to alter)
        self.show_legend = True
        self.grid(grid_on=True)

        self._xlabel = None
        self._ylabel = None
        self._yticks = None
        self._yticklabels = None
        self._xticks = None
        self._xticklabels = None
        # Limits have to applied after ticks to be effective
        self._ylim = None
        self._xlim = None

    def ylabel(self, label, *args, **kwargs):
        """Set a label for the y axis

        Parameters
        ----------
        label : string
            Label's text.
        *args, **kwargs
            Same as :meth:`.text` method.

        See Also
        --------
        .xlabel : Set a label for the x axis.

        """
        self._ylabel = label
        super(CPlotSecEval, self).ylabel(label, *args, **kwargs)

    def xlabel(self, label, *args, **kwargs):
        """Set a label for the x axis.

        Parameters
        ----------
        label : string
            Label's text.
        *args, **kwargs
            Same as :meth:`.text` method.

        Examples
        --------
        .. plot:: pyplots/xlabel.py
            :include-source:

        """
        self._xlabel = label
        super(CPlotSecEval, self).xlabel(label, *args, **kwargs)

    def yticks(self, location_array, *args, **kwargs):
        """Set the y-tick locations and labels.

        Parameters
        ----------
        location_array : CArray or list
            Contain ticks location.
        *args, **kwargs
            Same as :meth:`.text` method.

        See Also
        --------
        .xticks : Set the x-tick locations and labels.

        """
        self._yticks = location_array
        super(CPlotSecEval, self).yticks(location_array, *args, **kwargs)

    def yticklabels(self, labels, *args, **kwargs):
        """Set the ytick labels.

        Parameters
        ----------
        labels : list or CArray of string
            Xtick labels.
        *args, **kwargs
            Same as :meth:`.text` method.

        See Also
        --------
        .xticklabels : Set the xtick labels.

        """
        self._yticklabels = labels
        super(CPlotSecEval, self).yticklabels(labels, *args, **kwargs)

    def xticks(self, location_array, *args, **kwargs):
        """Set the x-tick locations and labels.

        Parameters
        ----------
        location_array : CArray or list
            Contain ticks location.
        *args, **kwargs
            Same as :meth:`.text` method.

        Examples
        --------
        .. plot:: pyplots/xticks.py
            :include-source:

        """
        self._xticks = location_array
        super(CPlotSecEval, self).xticks(location_array, *args, **kwargs)

    def xticklabels(self, labels, *args, **kwargs):
        """Set the xtick labels.

        Parameters
        ----------
        labels : list or CArray of string
            Xtick labels.
        *args, **kwargs
            Same as :meth:`.text` method.

        Examples
        --------
        .. plot:: pyplots/xticklabels.py
            :include-source:

        """
        self._xticklabels = labels
        super(CPlotSecEval, self).xticklabels(labels, *args, **kwargs)

    def ylim(self, bottom=None, top=None):
        """Set axes y limits.

        Parameters
        ----------
        bottom : scalar
            Starting value for the y axis.
        top : scalar
            Ending value for the y axis.

        See Also
        --------
        .xlim : Set x axis limits.

        """
        self._ylim = (bottom, top)
        super(CPlotSecEval, self).ylim(bottom=bottom, top=top)

    def xlim(self, bottom=None, top=None):
        """Set axes x limits.

        Parameters
        ----------
        bottom : scalar
            Starting value for the x axis.
        top : scalar
            Ending value for the x axis.

        Examples
        --------
        .. plot:: pyplots/xlim.py
            :include-source:

        """
        self._xlim = (bottom, top)
        super(CPlotSecEval, self).xlim(bottom=bottom, top=top)

    def _apply_params(self):
        """Apply defined parameters to active subplot."""
        fig_legend = self.get_legend()
        if fig_legend is not None:
            fig_legend.set_visible(self.show_legend)
        # Other axis parameters

        if self._ylabel:
            self.ylabel(self._ylabel)
        if self._xlabel:
            self.xlabel(self._xlabel)
        if self._yticks:
            self.yticks(self._yticks)
        if self._yticklabels:
            self.yticklabels(self._yticklabels)
        if self._xticks:
            self.xticks(self._xticks)
        if self._xticklabels:
            self.xticklabels(self._xticklabels)
        # Limits have to applied after ticks to be effective
        if self._ylim:
            self.ylim(*self._ylim)
        if self._xlim:
            self.xlim(*self._xlim)

    def _markers_idx(self, fp):
        """Returns the position of markers to plot.

        Parameters
        ----------
        fp : CArray
            False Positives.

        Returns
        -------
        ticks_idx : list
            List with the position of each xtick inside
            false positives array.

        Notes
        -----
        If a given xtick is not available inside `fp` array,
        the closest value's position will be returned.

        """
        return fp.binary_search(self._sp.get_xticks()).tolist()

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

        self._apply_params()

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
        self._apply_params()

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

        print sec_eval_data[0]
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
        for k in xrange(sngl_sec_eval_data.param_values.size):
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
