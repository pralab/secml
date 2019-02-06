from abc import ABCMeta

from scipy import linalg

from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradient
from secml.ml.classifiers.loss import CLoss
from secml.explanation import CExplainerLocal


# fixme: reckeck if everything is correct where there is a normalizer
# inside the classifier
class CExplainInfluence(CExplainerLocal):
    __metaclass__ = ABCMeta
    __super__ = "CExplainInfluence"

    def __init__(self, clf, tr, outer_loss_idx='logistic'):

        self._clf = clf
        self._tr = tr

        self._inv_H = None  # inverse hessian matrix
        self._grad_inner_loss_params = None

        self._outer_loss = CLoss.create(outer_loss_idx,
                                        extend_binary_labels=True)

        self._clf_gradient = CClassifierGradient.create(clf.class_type)

    def grad_outer_loss_params(self, x_ts, y_ts):
        """
        Compute derivate of the outer validation loss at test point(s) x
        This is typically not regularized (just an empirical loss function)
        :param x: a test point
        :param y: its label
        :return: dL_params, CArray of shape (d+1) * n_samples
        """
        f = self._clf.discriminant_function(x_ts)
        dl_df = self._outer_loss.dloss(y_ts, f)  # (n_samples,)
        df_dparams = self._clf_gradient.fd_params(x_ts, self._clf)  # (d+1) *
        # n_samples
        return df_dparams * dl_df.atleast_2d()

    def grad_inner_loss_params(self, x, y, clf):
        """
        Compute derivative of the inner training loss function
        for all training points.
        This is normally a regularized loss.
        :return:
        """
        return self._clf_gradient.Ld_params(x, y, clf)

    def hessian(self, x, y):
        """
        Compute hessian for the current parameters of the trained clf
        :param w:
        :return:
        """
        return self._clf_gradient.hessian(x, y, self._clf)

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

        d = x_ts.shape[1]
        H += 1e-9 * (CArray.eye(d + 1))

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
                self._tr.X, self._tr.Y, self._clf)

        v = self.grad_outer_loss_params(x_ts, y_ts).T.dot(self._inv_H).dot(
            self._grad_inner_loss_params)

        return v
