"""
.. module:: CClassifierGradientRidgeMixin
   :synopsis: Mixin for Ridge classifier gradients.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradientLinearMixin


class CClassifierGradientRidgeMixin(CClassifierGradientLinearMixin):
    """Mixin class for CClassifierRidge gradients."""

    # train derivatives:

    def hessian_tr_params(self, x, y=None):
        """Hessian of the training objective w.r.t. the classifier parameters.

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the training objective is computed.
        y : CArray
            Dataset labels.

        """
        alpha = self.alpha

        x = x.atleast_2d()
        n = x.shape[0]

        # handle normalizer, if present
        x = x if self.preprocess is None else self.preprocess.transform(x)

        d = x.shape[1]  # number of features in the normalized space

        H = CArray.zeros(shape=(d + 1, d + 1))
        Sigma = x.T.dot(x)
        dww = Sigma + alpha * CArray.eye(d)
        dwb = x.sum(axis=0)
        H[:-1, :-1] = dww
        H[-1, -1] = n  # + self.alpha
        H[-1, :-1] = dwb
        H[:-1, -1] = dwb.T
        H *= 2.0

        return H
