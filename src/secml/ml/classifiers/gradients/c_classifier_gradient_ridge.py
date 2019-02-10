"""
.. module:: CClassifierGradientSVM
   :synopsis: Class to compute the gradient of the SVM classifier

    @author: Battista Biggio
    @author: Ambra Demontis

"""

from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradientLinear
from secml.ml.classifiers.loss import CLossSquare
from secml.ml.classifiers.regularizer import CRegularizerL2


class CClassifierGradientRidge(CClassifierGradientLinear):
    class_type = 'ridge'

    def __init__(self):
        self._loss = CLossSquare()
        self._reg = CRegularizerL2()

    def _g(self, d):
        """
        d number of features
        """
        return CArray.eye(d)

    def _C(self, clf):
        return 1.0 / clf.alpha

    def hessian(self, clf, x, y=None):
        """
        Compute hessian for the current parameters of the trained clf
        """

        alpha = clf.alpha

        x = x.atleast_2d()
        n = x.shape[0]

        # handle normalizer, if present
        x = x if clf.preprocess is None else clf.preprocess.normalize(x)

        d = x.shape[1]  # number of features in the normalized space

        H = CArray.zeros(shape=(d + 1, d + 1))
        Sigma = (x.T).dot(x)
        dww = Sigma + alpha * self._g(d)
        dwb = x.sum(axis=0)
        H[:-1, :-1] = dww
        H[-1, -1] = n  # + clf.alpha
        H[-1, :-1] = dwb
        H[:-1, -1] = dwb.T
        H *= 2.0

        return H
