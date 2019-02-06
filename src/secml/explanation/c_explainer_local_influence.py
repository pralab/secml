"""
.. module:: CExplainerLocalInfluence
   :synopsis: Class to compute the Influence Function

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
.. moduleauthor:: Battista Biggio <battista.biggiodemontis@diee.unica.it>

"""

from abc import ABCMeta

from scipy import linalg

from secml.array import CArray
from secml.ml.classifiers.loss import CLoss
from secml.explanation import CExplainerLocal


class CExplainerLocalInfluence(CExplainerLocal):
    """
    Compute the influence function as it has been defined in:
    "Understanding Black-box Predictions via Influence Functions" by Koh et al.
    """
    __metaclass__ = ABCMeta
    __super__ = "CExplainInfluence"

    def __init__(self, clf, tr, outer_loss_idx='log'):

        self._clf = clf
        self._tr = tr

        self._inv_H = None  # inverse hessian matrix
        self._grad_inner_loss_params = None

        self._outer_loss = CLoss.create(outer_loss_idx)

    def grad_outer_loss_params(self, x_ts, y_ts):
        """
        Compute derivate of the outer validation loss at test point(s) x
        This is typically not regularized (just an empirical loss function)

        :param x: a test point
        :param y: its label
        :return: dL_params, CArray of shape (n_params +1 ) * n_samples
        """
        grad = self._clf.gradients.L_d_params(self._clf, x_ts, y_ts,
                                              loss=self._outer_loss,
                                              regularized=False)
        return grad

    def grad_inner_loss_params(self, x, y):
        """
        Compute derivative of the inner training loss function
        for all training points.
        This is normally a regularized loss.
        :return:
        """
        grad = self._clf.gradients.L_d_params(self._clf, x, y,
                                              loss=self._outer_loss,
                                              regularized=True)
        return grad

    def hessian(self, x, y):
        """
        Compute hessian for the current parameters of the trained clf
        :param w:
        :return:
        """
        return self._clf.gradients.hessian(self._clf, x, y)

    @property
    def clf(self):
        return self._clf

    @clf.setter
    def clf(self, value):
        self._clf = value
        self.__clear()

    @property
    def tr(self):
        return self._tr

    @tr.setter
    def tr(self, value):
        self._tr = value
        self.__clear()

    def __clear(self):
        """Reset the object."""
        self._grad_inner_loss_params = None

    def explain(self, x_ts, y_ts):
        """
        Compute influence of test sample x against all training samples
        :param x: the test sample
        :return: influence function values comparing x to all training samples
        """
        H = self.hessian(x_ts, y_ts)

        p = H.shape[0]
        H += 1e-9 * (CArray.eye(p))

        if self._inv_H is None:
            # compute hessian inverse
            det = linalg.det(H.tondarray())
            if abs(det) < 1e-6:
                self._inv_H = CArray(linalg.pinv2(H.tondarray()))
            else:
                self._inv_H = CArray(linalg.inv(H.tondarray()))

        x_ts = x_ts.atleast_2d()

        if self._grad_inner_loss_params is None:
            self._grad_inner_loss_params = self.grad_inner_loss_params(
                self._tr.X, self._tr.Y)

        v = self.grad_outer_loss_params(x_ts, y_ts).T.dot(self._inv_H).dot(
            self._grad_inner_loss_params)

        return v
