"""
.. module:: CPlotConfMatr
   :synopsis: Confusion matrix plots.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
import numpy as np
import itertools
from six.moves import range

from secml.figure._plots import CPlot


class CPlotConfMatr(CPlot):
    """Plots a Confusion Matrix."""

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
