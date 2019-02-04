from abc import ABCMeta, abstractmethod, abstractproperty

from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradient

class CClassifierGradientSVM(CClassifierGradient):

    class_type = 'svm'

    def _s(self, clf, tol=1e-6):
        """Indices of margin support vectors."""
        s = clf.alpha.find(
            (abs(clf.alpha) >= tol) *
            (abs(clf.alpha) <= clf.C - tol))
        return CArray(s)

    def _xs(self, clf):

        s = self._s(clf)

        if s.size == 0:
            return None

        xs = clf.sv[s, :].atleast_2d()
        return xs, s

    def hessian(self, clf):
        """
        Compute hessian for the current parameters of the trained clf
        :param w:
        :return:
        """
        svm = clf

        xs, sv_idx = self._xs(clf)  # these points are already normalized

        s = xs.shape[0]

        H = CArray.ones(shape=(s + 1, s + 1))
        H[:s, :s] = svm.kernel.k(xs)
        H[-1, -1] = 0

        return H

    def fd_params(self, x, y, clf):
        raise NotImplementedError()

    def Ld_params(self, x, y, clf):
        raise NotImplementedError()

    def Ld_s(self, x, y, clf):
        raise NotImplementedError()