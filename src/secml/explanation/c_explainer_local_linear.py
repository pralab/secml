"""
.. module:: CExplainerLocalLinear
   :synopsis: Local Explanation of predictions for linear classifiers.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.explanation import CExplainer
from secml.array import CArray
from secml.ml.classifiers import CClassifierLinear


class CExplainerLocalLinear(CExplainer):
    """Local Explanation of predictions for linear classifiers.

    In the cases of linear classifiers, the weights `w` give an explanation
    on the relevant features towards the prediction

    Parameters
    ----------
    clf : CClassifierLinear
        Instance of the classifier to explain. Must be linear.
    tr_ds : CDataset
        Training dataset of the classifier to explain.

    Attributes
    ----------
    class_type : 'linear'

    """
    __class_type = 'linear'

    def __init__(self, clf, tr_ds=None):

        if not isinstance(clf, CClassifierLinear) or not clf.is_linear():
            raise TypeError("input classifier must be linear.")

        super(CExplainerLocalLinear, self).__init__(clf, tr_ds)

    def explain(self, x):
        """Computes the explanation for input sample.

        Parameters
        ----------
        x : CArray
            Input sample.

        Returns
        -------
        attributions : CArray
            Attributions (weight of each feature) for input sample.

        """
        attr = self.clf.w.deepcopy()  # Attributions are just the weights
        self.logger.debug(
            "Attributions:\n{:}".format(attr))
        return attr
