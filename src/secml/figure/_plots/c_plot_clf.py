"""
.. module:: CPlotClassifier
   :synopsis: Classifier data plots.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from six.moves import range

import itertools

from sklearn.metrics import confusion_matrix

from secml.figure._plots import CPlot
from secml.array import CArray


class CPlotClassifier(CPlot):
    """Plots Classifier data.

    See Also
    --------
    .CPlot : basic subplot functions.
    .CFigure : creates and handle figures.

    """

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

        fmt = '%.2f' if normalize else 'd'

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
