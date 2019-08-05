"""
.. module:: CClassifierGradientLinearMixin
   :synopsis: Mixin for linear classifier gradients.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradientMixin
from secml.ml.classifiers.clf_utils import convert_binary_labels


class CClassifierGradientLinearMixin(CClassifierGradientMixin):
    """Mixin class for CClassifierLinear gradients."""

    # train derivatives:

    @staticmethod
    def _grad_f_w(x, d_l=None):
        """
        Derivative of the classifier decision function w.r.t. the
        weights
        """
        d = x.T  # where x is normalized if the classifier has a
        # normalizer
        if d_l is not None:
            d *= d_l
        return d

    @staticmethod
    def _grad_f_b(x, d_l=None):
        """Derivative of the classifier decision function w.r.t. the bias.

        Parameters
        ----------
        d_l : ??

        """
        # where x is normalized if the classifier has a normalizer
        x = x.atleast_2d()
        k = x.shape[0]  # number of samples
        d = CArray.ones((1, k))
        if d_l is not None:
            d *= d_l
        return d

    def grad_f_params(self, x, y=1):
        """Derivative of the decision function w.r.t. the classifier parameters.

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the training objective is computed.
        y : int
            Index of the class wrt the gradient must be computed.

        """
        if self.preprocess is not None:
            x = self.preprocess.transform(x)

        grad_f_w = self._grad_f_w(x)
        grad_f_b = self._grad_f_b(x)

        d = grad_f_w.append(grad_f_b, axis=0)

        return convert_binary_labels(y) * d

    def grad_loss_params(self, x, y, loss=None):
        """Derivative of the classifier loss w.r.t. the classifier parameters.

        d_loss / d_params = d_loss / d_f * d_f / d_params

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the loss is computed.
        y : CArray
            Dataset labels.
        loss: None (default) or CLoss
            If the loss is equal to None (default) the classifier loss is used
            to compute the derivative.

        """
        if loss is None:
            loss = self._loss

        y = y.ravel()

        C = self.C

        x = x.atleast_2d()

        s = self.decision_function(x)

        if self.preprocess is not None:
            x = self.preprocess.transform(x)

        d_l = loss.dloss(y, score=s).atleast_2d()
        grad_f_w = self._grad_f_w(x, d_l)  # d * n_samples
        grad_f_b = self._grad_f_b(x, d_l)  # 1 * n_samples

        grad_w = C * grad_f_w
        grad_b = C * grad_f_b

        grad = grad_w.append(grad_b, axis=0)

        return grad  # (d +1) * n_samples

    def grad_tr_params(self, x, y):
        """
        Derivative of the classifier training objective w.r.t. the classifier
         parameters.

        If the loss is equal to None (default) the classifier loss is used
        to compute the derivative.

        d_train_obj / d_params = d_loss / d_f * d_f / d_params + d_reg /
        d_params

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the loss is computed.
        y : CArray
            Dataset labels.

        """
        grad = self.grad_loss_params(x, y)

        w = CArray(self.w.ravel()).T  # column vector
        grad[:-1, :] += self._reg.dregularizer(w)

        return grad  # (d +1) * n_samples

    # test derivatives:

    def grad_f_x(self, x=None, y=1, **kwargs):
        """Computes the gradient of the classifier's output wrt input.

        Parameters
        ----------
        x : CArray
            The gradient is computed in the neighborhood of x.
        y : int, optional
            Index of the class wrt the gradient must be computed. Default 1.
        **kwargs
            Optional parameters for the function that computes the
            gradient of the decision function. See the description of
            each classifier for a complete list of optional parameters.

        Returns
        -------
        gradient : CArray
            Gradient of the classifier's output wrt input. Vector-like array.

        """
        return super(
            CClassifierGradientLinearMixin, self).grad_f_x(x, y, **kwargs)

    def _grad_f_x(self, x=None, y=1):
        """Computes the gradient of the linear classifier's decision function
         wrt decision function input.

        For linear classifiers, the gradient wrt input is equal
        to the weights vector w. The point x can be in fact ignored.

        Parameters
        ----------
        x : CArray or None, optional
            The gradient is computed in the neighborhood of x.
        y : int, optional
            Binary index of the class wrt the gradient must be computed.
            Default is 1, corresponding to the positive class.

        Returns
        -------
        gradient : CArray
            The gradient of the linear classifier's decision function
            wrt decision function input. Vector-like array.

        """
        # Gradient sign depends on input label (0/1)
        return convert_binary_labels(y) * self.w
