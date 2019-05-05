"""
.. module:: CClassifierGradientTestSVM
   :synopsis: Debugging class for the classifier gradient class

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""

from secml.array import CArray
from secml.ml.classifiers.gradients.tests.utils.gradient_test_classes import \
    CClassifierGradientTest


class CClassifierGradientTestSVM(CClassifierGradientTest):
    """
    This class implement different functionalities which are useful to test
    the CClassifierGradientSVM class.
    """
    __class_type = 'svm'

    def l(self, x, y, clf):
        """
        Classifier  loss
        """

        # compute the loss on the training samples
        scores = clf.decision_function(x)
        loss = clf._loss.loss(y, score=scores).atleast_2d()

        loss = clf.C * loss

        return loss

    def train_obj(self, x, y, clf):
        """
        Classifier training objective function
        """

        loss = self.l(x, y, clf)

        xs, margin_sv_idx = clf.sv_margin()
        Kss = clf.kernel.k(xs, xs)
        alpha_s = clf.alpha[margin_sv_idx]
        reg = alpha_s.atleast_2d().dot(Kss.dot(alpha_s.T))

        loss = clf.C * loss

        loss += reg

        return loss

    def params(self, clf):
        """
        Classifier parameters
        """
        margin_sv_idx = clf.sv_margin_idx()  # get the idx of the margin support vector
        return clf.alpha[margin_sv_idx].append(CArray(clf.b), axis=None)

    def change_params(self, params, clf):
        """
        Return a deepcopy of the given classifier with the value of the
        parameters changed
        vector
        """
        new_clf = clf.deepcopy()
        # get the idx of the margin support vector
        margin_sv_idx = clf.sv_margin_idx()
        new_clf._alpha[margin_sv_idx] = params[:-1]
        new_clf._b = params[-1]
        return new_clf
