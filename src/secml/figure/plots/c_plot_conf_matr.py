import numpy as np
import itertools
from six.moves import range

from secml.figure.plots import CPlot


class CPlotConfMatr(CPlot):
    """Confusion Matrix Plot.

    Attributes
    ----------
    class_type : 'conf-matrix'

    """
    __class_type = 'conf-matrix'

    def __init__(self, sp, default_params=None):
        # Calling CPlot constructor
        super(CPlotConfMatr, self).__init__(
            sp=sp, default_params=default_params)

        # Specific plot parameters (use `set_params` to alter)
        self.ylabel("True label")
        self.xlabel("Predicted label")

    def _apply_params(self):
        """Apply defined parameters to active subplot."""
        fig_legend = self.get_legend()
        if fig_legend is not None:
            fig_legend.set_visible(self.show_legend)

    def plot_confusion_matrix(
            self, cnf_matrix, title=None, normalize=True, cmap='jet'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.

        cnf_matrix : CArray
            Confusion matrix
        class_names: list
            List of class names
        title: String
            Plot title
        normalize: Boolean
            If true normalize the confusion matrix
        cmap: Colormap
        """

        img = self.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
        if title:
            self.title(title)
        self.colorbar(img)
        #tick_marks = np.arange(len(class_names))
        #self.xticks(tick_marks, class_names)  # rotatation 45
        #self.yticks(tick_marks, class_names)

        cnf_matrix = cnf_matrix.tondarray()
        if normalize:
            cnf_matrix = cnf_matrix.astype('float') / \
                         cnf_matrix.sum(axis=1)[:, np.newaxis]

        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(range(cnf_matrix.shape[0]),
                                      range(cnf_matrix.shape[1])):
            self.text(j, i, cnf_matrix[i, j].round(2),
                      horizontalalignment="center",
                      color="white" if cnf_matrix[i, j] > thresh else "black")

        # Customizing figure
        self._apply_params()
