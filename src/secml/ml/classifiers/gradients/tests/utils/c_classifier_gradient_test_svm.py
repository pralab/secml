"""
.. module:: CClassifierGradientTestSVM
   :synopsis: Debugging class for the classifier gradient class

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""

from secml.array import CArray
from secml.ml.classifiers.gradients.tests.utils import CClassifierGradientTest


class CClassifierGradientTestSVM(CClassifierGradientTest):
    """
    This class implement different functionalities which are useful to test
    the CClassifierGradientSVM class.
    """
    class_type = 'svm'

    def _L_tot(self, x, y, clf):
        """
        Classifier total loss
        L_tot = loss computed on the training samples + regularizer
        """

        # compute the loss on the training samples
        scores = clf.decision_function(x)
        loss = self._loss.loss(y, score=scores).atleast_2d()

        xs, margin_sv_idx = clf.xs()
        Kss = clf.kernel.k(xs, xs)
        alpha_s = clf.alpha[margin_sv_idx]
        reg = alpha_s.atleast_2d().dot(Kss.dot(alpha_s.T))

        loss = clf.C * loss + reg

        return loss

    def _params(self, clf):
        """
        Classifier parameters
        """
        margin_sv_idx = clf.s()  # get the idx of the margin support vector
        return clf.alpha[margin_sv_idx].append(CArray(clf.b), axis=None)

    def _change_params(self, params, clf):
        """
        Return a deepcopy of the given classifier with the value of the
        parameters changed
        vector
        """
        new_clf = clf.deepcopy()
        margin_sv_idx = clf.s()  # get the idx of the margin support vector
        new_clf._alpha[margin_sv_idx] = params[:-1]
        new_clf._b = params[-1]
        return new_clf
