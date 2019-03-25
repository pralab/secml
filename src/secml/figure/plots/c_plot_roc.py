from __future__ import division
from six.moves import range

from secml.figure.plots import CPlot
from secml.ml.peval.metrics import CRoc


class CPlotRoc(CPlot):
    """Plots the receiver operating characteristic curve, or ROC curve.

    This class creates a figure plotting in a fancy and standard
    style data from `.CRoc` class.

    Custom plotting parameters can be specified.
    Currently parameters default:
     - show_legend: True. Set False to hide legend on next plot.
     - ylabel: 'True Positive Rate (%)'.
     - xlabel: 'False Positive Rate (%)'.
     - yticks: [0, 20, 40, 60, 80, 100].
     - yticklabels: see yticks
     - xticks: list. [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100].
     - xticklabels: see xticks.
     - ylim: (0.1, 100).
     - xlim: (0, 100).
     - grid: True.

    Parameters
    ----------
    sp : Axes
        Subplot to use for plotting. Instance of `matplotlib.axes.Axes`.
    default_params : dict
        Dictionary with default parameters.

    Attributes
    ----------
    class_type : 'roc'

    See Also
    --------
    .CRoc : computes the receiver operating characteristic curve, or ROC curve.
    .CPlot : basic subplot functions.
    .CFigure : creates and handle figures.

    """
    __class_type = 'roc'

    def __init__(self, sp, default_params=None):

        # Calling CPlot constructor
        super(CPlotRoc, self).__init__(
            sp=sp, default_params=default_params)

        # Specific plot parameters (use `set_params` to alter)
        self.show_legend = True
        self.grid(grid_on=True)
        self.ylabel('True Positive Rate (%)')
        self.xlabel('False Positive Rate (%)')
        self.yticks([0, 20, 40, 60, 80, 100])
        self.yticklabels(['0', '20', '40', '60', '80', '100'])
        self.xticks([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100])
        self.xticklabels(['0.1', '0.5', '1', '2', '5', '10', '20', '50', '100'])
        # Limits have to applied after ticks to be effective
        self.ylim(0, 100)
        self.xlim(0.1, 100)

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
        super(CPlotRoc, self).ylabel(label, *args, **kwargs)

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
        super(CPlotRoc, self).xlabel(label, *args, **kwargs)

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
        super(CPlotRoc, self).yticks(location_array, *args, **kwargs)

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
        super(CPlotRoc, self).yticklabels(labels, *args, **kwargs)

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
        super(CPlotRoc, self).xticks(location_array, *args, **kwargs)

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
        super(CPlotRoc, self).xticklabels(labels, *args, **kwargs)

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
        super(CPlotRoc, self).ylim(bottom=bottom, top=top)

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
        super(CPlotRoc, self).xlim(bottom=bottom, top=top)

    def _apply_params(self):
        """Apply defined parameters to active subplot."""
        fig_legend = self.get_legend()
        if fig_legend is not None:
            fig_legend.set_visible(self.show_legend)
        # Other axis parameters
        self.ylabel(self._ylabel)
        self.xlabel(self._xlabel)
        self.yticks(self._yticks)
        self.yticklabels(self._yticklabels)
        self.xticks(self._xticks)
        self.xticklabels(self._xticklabels)
        # Limits have to applied after ticks to be effective
        self.ylim(*self._ylim)
        self.xlim(*self._xlim)

    def _markers_idx(self, fpr):
        """Returns the position of markers to plot.

        Parameters
        ----------
        fpr : CArray
            False Positive Rates.

        Returns
        -------
        ticks_idx : list
            List with the position of each xtick inside
            false positives array.

        Notes
        -----
        If a given xtick is not available inside `fpr` array,
        the closest value's position will be returned.

        """
        return fpr.binary_search(self._sp.get_xticks()).tolist()

    # TODO: REMOVE STYLE
    def plot_roc(self, fpr, tpr, label=None, style=None, logx=True):
        """Plot a ROC curve given input fpr and tpr.

        Curves will be plotted inside the active figure or
        a new figure will be created using default parameters.

        Parameters
        ----------
        fpr : CArray
            Array with False Positive Rats.
        tpr : CArray
            Array with False Positive Rates.
        label : str or None, optional
            Label to assign to the roc.
        style : str or None, optional
            Style of the roc plot.
        logx : bool, optional
            If True (default), logarithmic scale will be used for fpr axis.

        Returns
        -------
        roc_plot : CFigure
            Figure after this plot session.

        """
        if fpr.size != tpr.size:
            raise ValueError("input tpr and fpr arrays must have same length.")

        # TODO: REMOVE AFTER COLORMAPS ARE IMPLEMENTED IN CFIGURE
        styles = ['go-', 'yp--', 'rs-.', 'bD--', 'c-.', 'm-', 'y-.']

        plot_func = self.semilogx if logx is True else self.plot

        plot_func(fpr * 100, tpr * 100,
                  styles[self.n_lines % len(styles)] if style is None else style,
                  label=label, markevery=self._markers_idx(fpr * 100))

        if label is not None:
            # Legend on the lower right
            self.legend(loc=1, labelspacing=0.4, handletextpad=0.3)
        # Customizing figure
        self._apply_params()

    # TODO: REMOVE STYLE
    def plot_mean(self, roc, label=None, invert_tpr=False,
                  style=None, plot_std=False, logx=True):
        """Plot the mean of ROC curves.

        Curves will be plotted inside the active figure or
        a new figure will be created using default parameters.

        Parameters
        ----------
        roc : CRoc
            Roc curves to plot.
        label : str or None, optional
            Label to assign to the roc.
        invert_tpr : bool
            True if 1 - tpr (False Negative Rates) should be plotted
            on y axis. Default False.
        style : str or None, optional
            Style of the roc plot. If a string, must follow the
            matplotlib convention: '[color][marker][line]'.
        plot_std : bool (default False)
            If True, standard deviation of True Positive Rates will be plotted.
        logx : bool, optional
            If True (default), logarithmic scale will be used for fpr axis.

        Returns
        -------
        roc_plot : CFigure
            Figure after this plot session.

        """
        if not isinstance(roc, CRoc):
            raise TypeError("input must be a `CRoc` instance.")

        if roc.has_mean is False:
            raise ValueError("average for input roc has not been computed. "
                             "Use `CRoc.average()` first.")

        # TODO: REMOVE AFTER COLORMAPS ARE IMPLEMENTED IN CFIGURE
        styles = ['go-', 'yp--', 'rs-.', 'bD--', 'c-.', 'm-', 'y-.']

        # If std should be plotted each run plots 2 curvers
        n_lines = int(self.n_lines / 2) if plot_std is True else self.n_lines
        # Get indices of fpr @ xticks
        mkrs_idx = self._markers_idx(roc.mean_fpr * 100)

        mean_tpr = roc.mean_tpr if invert_tpr is False else 1 - roc.mean_tpr
        plot_func = self.semilogx if logx is True else self.plot
        plot_func(roc.mean_fpr * 100, mean_tpr * 100,
                  styles[n_lines % len(styles)] if style is None else style,
                  label=label, markevery=mkrs_idx)

        if plot_std is True:
            if roc.has_std_dev is False:
                raise ValueError("roc object has no standard deviation for data.")
            self.errorbar(roc.mean_fpr[mkrs_idx] * 100, mean_tpr[mkrs_idx] * 100,
                          ecolor=styles[n_lines % len(styles)][0] if style is None else style[0],
                          fmt='None', yerr=roc.std_dev_tpr[mkrs_idx] * 100)

        if label is not None:
            # Legend on the lower right
            self.legend(loc=4 if invert_tpr is False else 1,
                        labelspacing=0.4, handletextpad=0.3)
        # Customizing figure
        self._apply_params()

    def plot_repetitions(self, roc, label=None, invert_tpr=False, logx=True):
        """Plot all input ROC curves.

        Curves will be plotted inside the active figure or
        a new figure will be created using default parameters.

        Parameters
        ----------
        roc : CRoc
            Roc curves to plot.
        label : str or None, optional
            Label to assign to the roc.
            Repetition number will be appended using the
            following convention:
             - If label is None -> "rep 'i'"
             - If label is not None -> "`label` (rep `i`)"
        invert_tpr : bool
            True if 1 - tpr (False Negative Rates) should be plotted
            on y axis. Default False.
        logx : bool, optional
            If True (default), logarithmic scale will be used for fpr axis.

        Returns
        -------
        roc_plot : CFigure
            Figure after this plot session.

        """
        def label_w_rep(l_str, i):
            """Format input label to show repetition number.

            Parameters
            ----------
            l_str : str
                Original label.
            i : int
                Repetition index number.

            Returns
            -------
            out_label : str
                Label formatted as following:
                1) If label is '' -> "rep 'i'"
                2) If label is not '' -> "`label` (rep `i`)"

            """
            i_label = 'rep {:}'.format(i)
            if l_str is not None:
                i_label = l_str + ' (' + i_label + ')'

            return i_label

        if not isinstance(roc, CRoc):
            raise TypeError("input must be a `CRoc` instance.")

        # TODO: REMOVE AFTER COLORMAPS ARE IMPLEMENTED IN CFIGURE
        styles = ['go-', 'yp--', 'rs-.', 'bD--', 'c-.', 'm-', 'y-.']
        # Storing number of lines already plotted to chose style accordingly
        n_lines = self.n_lines

        plot_func = self.semilogx if logx is True else self.plot

        for rep_i in range(roc.n_reps):

            if roc.n_reps <= 1:  # For one rep ROC is stored as CArray
                tpr = roc.tpr
                fpr = roc.fpr
            else:  # For more than one rep ROC is stored as lists
                tpr = roc.tpr[rep_i]
                fpr = roc.fpr[rep_i]

            tpr = tpr if invert_tpr is False else 1 - tpr

            plot_func(fpr * 100, tpr * 100,
                      styles[(n_lines + rep_i) % len(styles)],
                      label=label_w_rep(label, rep_i),
                      markevery=self._markers_idx(fpr * 100))

        if label is not None:
            # Legend on the lower right
            self.legend(loc=4 if invert_tpr is False else 1,
                        labelspacing=0.4, handletextpad=0.3)

        # Customizing figure
        self._apply_params()


