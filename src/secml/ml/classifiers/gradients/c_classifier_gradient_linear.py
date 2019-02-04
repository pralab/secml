"""
.. module:: CClassifierGradientLinear
   :synopsis: Common interface for the implementations of linear classifier
   gradients

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
"""

from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradient
from secml.ml.classifiers.clf_utils import convert_binary_labels


class CClassifierGradientLinear(CClassifierGradient):
    class_type = 'grad_lin'

    def fd_w(self, x):
        """
        Derivative of the discriminant function w.r.t. the classifier
        weights
        """
        d = x.T  # where x is normalized if the classifier has a
        # normalizer
        return d

    def fd_b(self, x):
        """
        Derivative of the discriminant function w.r.t. the classifier
        bias
        """
        x = x.atleast_2d()  # where x is normalized if the classifier has a
        # normalizer
        k = x.shape[0]  # number of samples
        d = CArray.ones((1, k))
        return d

    def fd_params(self, x, clf):
        """
        Derivative of the discriminant function w.r.t. the classifier
        parameters
        """
        if clf.normalizer is not None:
            x = clf.normalizer.normalize(x)

        fd_w = self.fd_w(x)
        fd_b = self.fd_b(x)
        d = fd_w.append(fd_b, axis=0)
        return d

    def fd_x(self, x=None, y=1):
        """
        Derivative of the discriminant function w.r.t. an input sample
        """
        # Gradient sign depends on input label (0/1)
        return convert_binary_labels(y) * self.w

    def dreg_s(self, w):
        """
        Derivative of the regularizer w.r.t. the score
        """
        return self._reg.dregularizer(w)

    def Ld_params(self, x, y, clf):
        """
        Derivative of the classifier classifier loss function (regularizer
        included) w.r.t. the classifier parameters

        dL / d_params = dL / df * df / d_params + dReg / d_params
        """

        y = y.ravel()

        w = CArray(clf.w.ravel()).T  # column vector
        C = clf.C

        x = x.atleast_2d()

        s = clf.decision_function(x)

        fd_w = self.fd_w(x)  # d * n_samples
        fd_b = self.fd_b(x)  # 1 * n_samples

        grad_w = C * (self._loss.dloss(y=y, score=s).atleast_2d() * fd_w) + \
                 self._reg.dregularizer(w)
        grad_b = C * (self._loss.dloss(y=y, score=s).atleast_2d() * fd_b)

        grad = grad_w.append(grad_b, axis=0)

        return grad  # (d +1) * n_samples

    def Ld_s(self, w, y, score):
        """
        Derivative of the classifier loss function w.r.t. the score
        """
        return self._loss.dloss(y, score)
