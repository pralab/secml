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

    # test derivatives:

    def _grad_f_x(self, x=None, y=1):
        """Computes the gradient of the SVM classifier's decision function
         wrt decision function input.

        If the SVM classifier is linear, the gradient wrt input is equal
        to the weights vector w. The point x can be in fact ignored.

        Otherwise, for non-linear SVM, the gradient is computed
        in the dual representation:

        .. math::

            \sum_i y_i alpha_i \diff{K(x,xi)}{x}

        Parameters
        ----------
        x : CArray or None, optional
            The gradient is computed in the neighborhood of x.
            For non-linear classifiers, x is required.
        y : int, optional
            Binary index of the class wrt the gradient must be computed.
            Default is 1, corresponding to the positive class.

        Returns
        -------
        gradient : CArray
            The gradient of the SVM classifier's decision function
            wrt decision function input. Vector-like array.

        """
        if self.is_kernel_linear():  # Simply return w for a linear SVM
            return CClassifierGradientLinearMixin._grad_f_x(self, y=y)

        # Point is required in the case of non-linear SVM
        if x is None:
            raise ValueError("point 'x' is required to compute the gradient")

        # TODO: ADD OPTION FOR RANDOM SUBSAMPLING OF SVs
        # Gradient in dual representation: \sum_i y_i alpha_i \diff{K(x,xi)}{x}
        m = int(self.grad_sampling * self.n_sv.sum())  # Equivalent to floor
        idx = CArray.randsample(self.alpha.size, m)  # adding some randomness

        gradient = self.kernel.gradient(self.sv[idx, :], x).atleast_2d()

        # Few shape check to ensure broadcasting works correctly
        if gradient.shape != (idx.size, self.n_features):
            raise ValueError("Gradient shape must be ({:}, {:})".format(
                idx.size, self.n_features))

        alpha_2d = self.alpha[idx].atleast_2d()
        if gradient.issparse is True:  # To ensure the sparse dot is used
            alpha_2d = alpha_2d.tosparse()
        if alpha_2d.shape != (1, idx.size):
            raise ValueError(
                "Alpha vector shape must be "
                "({:}, {:}) or ravel equivalent".format(1, idx.size))

        gradient = alpha_2d.dot(gradient)

        # Gradient sign depends on input label (0/1)
        return convert_binary_labels(y) * gradient.ravel()
