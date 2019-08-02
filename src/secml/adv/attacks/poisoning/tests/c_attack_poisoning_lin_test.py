"""
.. module:: CAttackPoisoningLinTest
   :synopsis: Debugging class for poisoning against linear classifiers

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml.core import CCreator


class CAttackPoisoningLinTest(CCreator):
    """
    This class implement different functionalities which are useful to test
    the gradients which are needed to poisoning a linear classifier.
    """

    def __init__(self, pois_obj):
        self.pois_obj = pois_obj

    def w1(self, xc):
        """
        Parameters
        ----------
        xc: poisoning point

        Returns
        -------
        f_obj: values of objective function (average hinge loss) at x
        """
        idx, clf, tr = self._clf_poisoning(xc)

        return clf.w.ravel()[0]

    def w2(self, xc):
        """
        Parameters
        ----------
        xc: poisoning point

        Returns
        -------
        f_obj: values of objective function (average hinge loss) at x
        """

        idx, clf, tr = self._clf_poisoning(xc)

        return clf.w.ravel()[1]

    def b(self, xc):
        """
        Parameters
        ----------
        xc: poisoning point

        Returns
        -------
        f_obj: values of objective function (average hinge loss) at x
        """

        idx, clf, tr = self._clf_poisoning(xc)

        return clf.b

    def _clf_poisoning(self, xc):

        xc = xc.atleast_2d()
        n_samples = xc.shape[0]

        if n_samples > 1:
            raise TypeError("x is not a single sample!")

        # index of poisoning point within xc.
        # This will be replaced by the input parameter xc
        if self.pois_obj._idx is None:
            idx = 0
        else:
            idx = self.pois_obj._idx

        self.pois_obj._xc[idx, :] = xc
        clf, tr = self.pois_obj._update_poisoned_clf()

        return idx, clf, tr

    def _preparation_for_grad_computation(self, xc):

        idx, clf, tr = self._clf_poisoning(xc)

        y_ts = self.pois_obj._y_target if self.pois_obj._y_target is not \
                                          None else self.pois_obj.val.Y

        # computing gradient of loss(y, f(x)) w.r.t. f
        score = clf.decision_function(self.pois_obj.val.X)
        loss_grad = self.pois_obj._attacker_loss.dloss(y_ts, score)

        return idx, clf, loss_grad, tr

    def _grads_computation(self, xc):
        """
        Compute the derivative of the classifier parameters w.r.t. the
        poisoning points xc.

        The result is a CArray of dimension d * (d+1) where d is equal to the
        number of features

        """
        idx, clf, loss_grad, tr = self._preparation_for_grad_computation(xc)
        self.pois_obj._gradient_fk_xc(self.pois_obj._xc[idx, :],
                                      self.pois_obj._yc[idx],
                                      clf, loss_grad, tr)
        grads = self.pois_obj._d_params_xc
        return grads

    def gradient_w1_xc(self, xc):

        grad = self._grads_computation(xc)

        norm = grad.norm_2d()
        grad = grad / norm if norm > 0 else grad

        return grad[:, 0].ravel()

    def gradient_w2_xc(self, xc):

        grad = self._grads_computation(xc)

        norm = grad.norm_2d()
        grad = grad / norm if norm > 0 else grad

        return grad[:, 1].ravel()

    def gradient_b_xc(self, xc):

        grad = self._grads_computation(xc)

        norm = grad.norm_2d()
        grad = grad / norm if norm > 0 else grad

        return grad[:, 2].ravel()
