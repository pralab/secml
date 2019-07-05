"""
.. module:: CExplainerLocalInfluence
   :synopsis: Class to compute the Influence Function

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
.. moduleauthor:: Battista Biggio <battista.biggiodemontis@diee.unica.it>

"""
from scipy import linalg

from secml.array import CArray
from secml.ml.classifiers.loss import CLoss
from secml.explanation import CExplainerLocal


class CExplainerLocalInfluence(CExplainerLocal):
    """
    Compute the influence function as it has been defined in:
    "Understanding Black-box Predictions via Influence Functions" by Koh et al.
    
    Attributes
    ----------
    class_type : 'influence'
    
    """
    __class_type = 'influence'

    def __init__(self, clf, tr_ds, outer_loss_idx='log'):

        super(CExplainerLocalInfluence, self).__init__(clf=clf, tr_ds=tr_ds)

        self._inv_H = None  # inverse hessian matrix
        self._grad_inner_loss_params = None

        self._outer_loss = CLoss.create(outer_loss_idx)

    def grad_outer_loss_params(self, x, y):
        """
        Compute derivate of the outer validation loss at test point(s) x
        This is typically not regularized (just an empirical loss function)

        :param x: a test point
        :param y: its label
        :return: dL_params, CArray of shape (n_params +1 ) * n_samples
        """
        # FIXME: this is the validation loss. Why are we calling the clf?
        grad = self.clf.grad_loss_params(x, y)
        return grad

    def grad_inner_loss_params(self, x, y):
        """
        Compute derivative of the inner training loss function
        for all training points.
        This is normally a regularized loss.
        :return:
        """
        grad = self.clf.grad_tr_params(x, y)
        return grad

    def hessian(self, x, y):
        """
        Compute hessian for the current parameters of the trained clf
        :param w:
        :return:
        """
        return self.clf.hessian_tr_params(x, y)

    def explain(self, x, y):
        """
        Compute influence of test sample x against all training samples
        :param x: the test sample
        :return: influence function values comparing x to all training samples
        """
        H = self.hessian(x, y)

        p = H.shape[0]
        H += 1e-9 * (CArray.eye(p))

        if self._inv_H is None:
            # compute hessian inverse
            det = linalg.det(H.tondarray())
            if abs(det) < 1e-6:
                self._inv_H = CArray(linalg.pinv2(H.tondarray()))
            else:
                self._inv_H = CArray(linalg.inv(H.tondarray()))

        x = x.atleast_2d()

        if self._grad_inner_loss_params is None:
            self._grad_inner_loss_params = self.grad_inner_loss_params(
                self.tr_ds.X, self.tr_ds.Y)

        v = self.grad_outer_loss_params(x, y).T.dot(self._inv_H).dot(
            self._grad_inner_loss_params)

        return v
