"""
.. module:: CClassifierGradientTestLinear
   :synopsis: Debugging class for the classifier gradient class

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""

from secml.array import CArray
from secml.ml.classifiers.gradients.tests.utils import CClassifierGradientTest


class CClassifierGradientTestLinear(CClassifierGradientTest):
    """
    This class implement different functionalities which are useful to test
    the CClassifierGradientSVM class.
    """

    def L_tot(self, x, y, clf):
        """
        Classifier total loss
        L_tot = loss computed on the training samples + regularizer
        """
        y = y.ravel()

        w = CArray(clf.w.ravel()).T  # column vector
        C = self.gradients._C(clf)

        x = x.atleast_2d()

        s = clf.decision_function(x)

        loss = C * self.gradients._loss.loss(y,
                                             score=s) + self.gradients._reg.regularizer(
            w)

        return loss

    def params(self, clf):
        """
        Classifier parameters
        """
        return clf.w.append(CArray(clf.b), axis=None)

    def change_params(self, params, clf):
        """
        Return a deepcopy of the given classifier with the value of the
        parameters changed
        vector
        """
        new_clf = clf.deepcopy()
        new_clf._w = params[:-1]
        new_clf._b = params[-1]
        return new_clf
