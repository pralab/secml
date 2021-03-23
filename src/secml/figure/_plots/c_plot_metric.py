"""
.. module:: CPlotMetric
   :synopsis: Performance evaluation metrics plots.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
import itertools

from sklearn.metrics import confusion_matrix

from secml.figure._plots import CPlot
from secml.ml.peval.metrics import CRoc
from secml.array import CArray


class CPlotMetric(CPlot):
    """Plots of performance evaluation metrics.

    Currently parameters default for ROC plots:
     - show_legend: True
     - ylabel: 'True Positive Rate (%)'
     - xlabel: 'False Positive Rate (%)'
     - yticks: [0, 20, 40, 60, 80, 100]
     - yticklabels: see yticks
     - xticks: list. [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
     - xticklabels: see xticks
     - ylim: (0.1, 100)
     - xlim: (0, 100)
     - grid: True

    See Also
    --------
    .CPlot : basic subplot functions.
    .CFigure : creates and handle figures.

    """

    def apply_params_roc(self):
        """Apply defined parameters to active subplot."""
        fig_legend = self.get_legend()
        if self.show_legend is not False and fig_legend is not None:
            fig_legend.set_visible(True)
        self.grid(grid_on=True)
        if self._ylabel is None:
            self.ylabel('True Positive Rate (%)')
        if self._xlabel is None:
            self.xlabel('False Positive Rate (%)')
        if self._yticks is None:
            self.yticks([0, 20, 40, 60, 80, 100])
        if self._yticklabels is None:
            self.yticklabels(['0', '20', '40', '60', '80', '100'])
        if self._xticks is None:
            self.xticks([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100])
        if self._xticklabels is None:
            self.xticklabels(['0.1', '0.5', '1', '2', '5', '10', '20', '50', '100'])
        # Limits have to applied after ticks to be effective
        if self._ylim is None:
            self.ylim(0, 100)
        if self._xlim is None:
            self.xlim(0.1, 100)

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

        # Customizing figure
        self.apply_params_roc()

        # TODO: REMOVE AFTER COLORMAPS ARE IMPLEMENTED IN CFIGURE
        styles = ['go-', 'yp--', 'rs-.', 'bD--', 'c-.', 'm-', 'y-.']

        plot_func = self.semilogx if logx is True else self.plot

        plot_func(fpr * 100, tpr * 100,
                  styles[self.n_lines % len(styles)] if style is None else style,
                  label=label, markevery=self.get_xticks_idx(fpr * 100))

        if label is not None:
            # Legend on the lower right
            self.legend(loc=1, labelspacing=0.4, handletextpad=0.3)

        if logx is True:  # xticks have been reset by semilogx, reassign them
            self.xticks(self._xticks)
            self.xticklabels(self._xticklabels)

    # TODO: REMOVE STYLE
    def plot_roc_mean(self, roc, label=None, invert_tpr=False,
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

        # Customizing figure
        self.apply_params_roc()

        # TODO: REMOVE AFTER COLORMAPS ARE IMPLEMENTED IN CFIGURE
        styles = ['go-', 'yp--', 'rs-.', 'bD--', 'c-.', 'm-', 'y-.']

        # If std should be plotted each run plots 2 curvers
        n_lines = int(self.n_lines / 2) if plot_std is True else self.n_lines
        # Get indices of fpr @ xticks
        mkrs_idx = self.get_xticks_idx(roc.mean_fpr * 100)

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

        if logx is True:  # xticks have been reset by semilogx, reassign them
            self.xticks(self._xticks)
            self.xticklabels(self._xticklabels)

    def plot_roc_reps(self, roc, label=None, invert_tpr=False, logx=True):
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

        # Customizing figure
        self.apply_params_roc()

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
                      markevery=self.get_xticks_idx(fpr * 100))

        if label is not None:
            # Legend on the lower right
            self.legend(loc=4 if invert_tpr is False else 1,
                        labelspacing=0.4, handletextpad=0.3)

        if logx is True:  # xticks have been reset by semilogx, reassign them
            self.xticks(self._xticks)
            self.xticklabels(self._xticklabels)

    # FIXME: accept a CMetricConfusionMatrix object instead
    def plot_confusion_matrix(self, y_true, y_pred,
                              normalize=False, labels=None,
                              title=None, cmap='Blues', colorbar=False):
        """Plot a confusion matrix.

        y_true : CArray
            True labels.
        y_pred : CArray
            Predicted labels.
        normalize : bool, optional
            If True, normalize the confusion matrix in 0/1. Default False.
        labels : list, optional
            List with the label of each class.
        title: str, optional
            Title of the plot. Default None.
        cmap: str or matplotlib.pyplot.cm, optional
            Colormap to use for plotting. Default 'Blues'.
        colorbar : bool, optional
            If True, show the colorbar side of the matrix. Default False.

        """
        matrix = CArray(confusion_matrix(
            y_true.tondarray(), y_pred.tondarray()))

        if normalize:  # min-max normalization
            matrix_min = matrix.min()
            matrix_max = matrix.max()
            matrix = (matrix - matrix.min()) / (matrix_max - matrix_min)

        ax = self.imshow(matrix, interpolation='nearest', cmap=cmap)

        self._sp.set_xticks(CArray.arange(matrix.shape[1]).tondarray())
        self._sp.set_yticks(CArray.arange(matrix.shape[0]).tondarray())
        if labels is not None:
            self._sp.set_xticklabels(labels)
            self._sp.set_yticklabels(labels)

        # Rotate the tick labels and set their alignment.
        import matplotlib.pyplot as plt
        plt.setp(self._sp.get_xticklabels(), rotation=45,
                 ha="right", rotation_mode="anchor")

        fmt = '.2f' if normalize else 'd'

        if colorbar is True:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="5%", pad=0.1)
            # TODO: set format -> cax.set_yticklabels
            self.colorbar(ax, cax=cax)

        if title is True:
            self.title(title)

        thresh = matrix.max() / 2.
        for i, j in itertools.product(
                range(matrix.shape[0]), range(matrix.shape[1])):
            self.text(j, i, format(matrix[i, j].item(), fmt),
                      horizontalalignment="center",
                      color="white" if matrix[i, j] > thresh else "black")
