"""
.. module:: CClassifierGradientSVMMixin
   :synopsis: Mixin for SVM classifier gradients.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradientLinearMixin
from secml.ml.classifiers.clf_utils import convert_binary_labels


class CClassifierGradientSVMMixin(CClassifierGradientLinearMixin):
    """Mixin class for CClassifierSVM gradients."""

    # train derivatives:

    def hessian_tr_params(self, x=None, y=None):
        """
        Hessian of the training objective w.r.t. the classifier parameters.
        """
        xs, sv_idx = self.sv_margin()  # these points are already normalized

        s = xs.shape[0]

        H = CArray.ones(shape=(s + 1, s + 1))
        H[:s, :s] = self.kernel.k(xs)
        H[-1, -1] = 0

        return H

    def grad_f_params(self, x, y=1):
        """Derivative of the decision function w.r.t. the classifier parameters.

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the training objective is computed.
        y : int
            Index of the class wrt the gradient must be computed.

        """
        xs, sv_idx = self.sv_margin()  # these points are already normalized

        if xs is None:
            self.logger.debug("Warning: sv_margin is empty "
                              "(all points are error vectors).")
            return None

        xk = x if self.preprocess is None else self.preprocess.transform(x)

        s = xs.shape[0]  # margin support vector
        k = xk.shape[0]

        Ksk_ext = CArray.ones(shape=(s + 1, k))
        Ksk_ext[:s, :] = self.kernel.k(xs, xk)

        return convert_binary_labels(y) * Ksk_ext  # (s + 1) * k

    def grad_loss_params(self, x, y, loss=None):
        """
        Derivative of the loss w.r.t. the classifier parameters

        dL / d_params = dL / df * df / d_params

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the loss is computed.
        y : CArray
            Labels of the training samples.
        loss: None (default) or CLoss
            If the loss is equal to None (default) the classifier loss is used
            to compute the derivative.

        """
        if loss is None:
            loss = self._loss

        # compute the loss derivative w.r.t. alpha
        f_params = self.grad_f_params(x)  # (s + 1) * n_samples
        scores = self.decision_function(x)

        dL_s = loss.dloss(y, score=scores).atleast_2d()
        dL_params = dL_s * f_params  # (s + 1) * n_samples

        grad = self.C * dL_params

        return grad

    def grad_tr_params(self, x, y):
        """Derivative of the classifier training objective w.r.t.
        the classifier parameters.

        dL / d_params = dL / df * df / d_params + dReg / d_params

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the loss is computed.
        y : CArray
            Features of the training samples

        """
        grad = self.grad_loss_params(x, y)

        # compute the regularizer derivative w.r.t alpha
        xs, margin_sv_idx = self.sv_margin()
        K = self.kernel.k(xs, xs)
        d_reg = 2 * K.dot(self.alpha[margin_sv_idx].T)  # s * 1

        # add the regularizer to the gradient of the alphas
        s = margin_sv_idx.size
        grad[:s, :] += d_reg

        return grad  # (s +1) * n_samples
