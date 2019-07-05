"""
.. module:: ClassifierGradientRidgeMixin
   :synopsis: Class to compute the gradient of the Ridge classifier

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradientLinearMixin
from secml.ml.classifiers.loss import CLossSquare
from secml.ml.classifiers.regularizer import CRegularizerL2

from secml.ml.classifiers.clf_utils import convert_binary_labels

from abc import abstractmethod


class CClassifierGradientRidgeMixin(CClassifierGradientLinearMixin):
    __class_type = 'ridge'

    def __init__(self):
        self._loss = CLossSquare()
        self._reg = CRegularizerL2()

        super(CClassifierGradientRidgeMixin, self).__init__()

    @property
    def C(self):
        return 1.0 / self.alpha

    # required classifier properties:

    @property
    @abstractmethod
    def alpha(self):
        pass

    # train derivatives:

    @staticmethod
    def _g(d):
        """
        d number of features
        """
        return CArray.eye(d)

    def hessian_tr_params(self, x, y=None):
        """
        Hessian of the training objective w.r.t. the classifier
        parameters

        Parameters
        ----------
        x : CArray
            features of the dataset on which the training objective is computed
        y :  CArray
            dataset labels
        """

        alpha = self.alpha

        x = x.atleast_2d()
        n = x.shape[0]

        # handle normalizer, if present
        x = x if self.preprocess is None else self.preprocess.transform(x)

        d = x.shape[1]  # number of features in the normalized space

        H = CArray.zeros(shape=(d + 1, d + 1))
        Sigma = (x.T).dot(x)
        dww = Sigma + alpha * self._g(d)
        dwb = x.sum(axis=0)
        H[:-1, :-1] = dww
        H[-1, -1] = n  # + self.alpha
        H[-1, :-1] = dwb
        H[:-1, -1] = dwb.T
        H *= 2.0

        return H

    # test derivatives:

    # fixme: remove as soon as we will have removed the kernel from the ridge
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
        x = x.atleast_2d()

        if self.is_kernel_linear():  # Simply return w for a linear Ridge
            return CClassifierGradientLinearMixin._grad_f_x(self, y=y)

        # Point is required in the case of non-linear Ridge
        if x is None:
            raise ValueError("point 'x' is required to compute the gradient")

        gradient = self.kernel.gradient(self._tr, x).atleast_2d()

        # Few shape check to ensure broadcasting works correctly
        if gradient.shape != (self._tr.shape[0], self.n_features):
            raise ValueError("Gradient shape must be ({:}, {:})".format(
                x.shape[0], self.n_features))

        w_2d = self.w.atleast_2d()
        if gradient.issparse is True:  # To ensure the sparse dot is used
            w_2d = w_2d.tosparse()
        if w_2d.shape != (1, self._tr.shape[0]):
            raise ValueError("Weight vector shape must be ({:}, {:}) "
                             "or ravel equivalent".format(1,
                                                          self._tr.shape[0]))

        gradient = w_2d.dot(gradient)

        # Gradient sign depends on input label (0/1)
        return convert_binary_labels(y) * gradient.ravel()
