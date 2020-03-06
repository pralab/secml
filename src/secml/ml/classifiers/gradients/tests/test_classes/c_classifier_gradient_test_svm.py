"""
.. module:: CClassifierGradientTestSVM
   :synopsis: Debugging class for mixin classifier gradient SVM.

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.ml.classifiers.gradients.tests.test_classes import \
    CClassifierGradientTest

from secml.array import CArray


class CClassifierGradientTestSVM(CClassifierGradientTest):
    __class_type = 'svm'

    def params(self, clf):
        """Classifier parameters."""
        return clf.alpha[clf.sv_margin_idx()].append(CArray(clf.b), axis=None)

    def l(self, x, y, clf):
        """Classifier loss."""
        loss = clf._loss.loss(y, score=clf.decision_function(x)).atleast_2d()
        return clf.C * loss

    def train_obj(self, x, y, clf):
        """Classifier training objective function."""
        loss = self.l(x, y, clf)

        xs, margin_sv_idx = clf.sv_margin()

        alpha_s = clf.alpha[margin_sv_idx]
        reg = alpha_s.atleast_2d().dot(clf.kernel.k(xs, xs).dot(alpha_s.T))

        return clf.C * loss + reg

    def change_params(self, params, clf):
        """Return a deepcopy of the given classifier with the value
        of the parameters changed."""
        new_clf = clf.deepcopy()
        new_clf._alpha[clf.sv_margin_idx()] = params[:-1]
        new_clf._b = params[-1]
        return new_clf
