"""
.. module:: CClassifierGradientTestLinear
   :synopsis: Debugging class for mixin classifier gradient linear.

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.ml.classifiers.gradients.tests.test_classes import \
    CClassifierGradientTest

from secml.array import CArray


class CClassifierGradientTestLinear(CClassifierGradientTest):
    __class_type = 'linear'

    def params(self, clf):
        """Classifier parameters."""
        return clf.w.append(CArray(clf.b), axis=None)

    def l(self, x, y, clf):
        """Classifier loss."""
        s = clf.decision_function(x.atleast_2d())
        return clf.C * clf._loss.loss(y.ravel(), score=s)

    def train_obj(self, x, y, clf):
        """Classifier training objective function."""
        return self.l(x, y, clf) + clf._reg.regularizer(clf.w)

    def change_params(self, params, clf):
        """Return a deepcopy of the given classifier with the value
        of the parameters changed."""
        new_clf = clf.deepcopy()
        # modifying internal variables w, b from sklearn models
        new_clf._sklearn_model.coef_ = params[:-1].tondarray()
        new_clf._sklearn_model.intercept_ = params[-1].tondarray()
        return new_clf
