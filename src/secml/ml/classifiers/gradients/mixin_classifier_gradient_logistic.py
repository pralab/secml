"""
.. module:: CClassifierGradientLogisticMixin
   :synopsis: Mixin for logistic classifier gradients.

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.array import CArray
from secml.ml.classifiers.gradients import CClassifierGradientLinearMixin
from secml.ml.classifiers.clf_utils import convert_binary_labels


class CClassifierGradientLogisticMixin(CClassifierGradientLinearMixin):
    """Mixin class for CClassifierLogistic gradients."""

    # train derivatives:

    def _sigm(self, y, s):
        """Sigmoid function."""
        y = CArray(y)
        s = CArray(s)
        return 1.0 / (1.0 + (-y * s).exp())

    def hessian_tr_params(self, x, y):
        """Hessian of the training objective w.r.t. the classifier parameters.

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the training objective is computed.
        y : CArray
            Dataset labels.

        """
        y = y.ravel()
        y = convert_binary_labels(y)
        y = CArray(y).astype(float).T  # column vector

        C = self.C

        x = x.atleast_2d()
        n = x.shape[0]

        # nb: we compute the score before the x normalization as decision
        # function normalizes x
        s = self.decision_function(x, y=1).T
        sigm = self._sigm(y, s)
        z = sigm * (1 - sigm)

        # handle normalizer, if present
        x = x if self.preprocess is None else self.preprocess.transform(x)

        d = x.shape[1]  # number of features in the normalized space

        # first derivative wrt b derived w.r.t. w
        diag = z * CArray.eye(n_rows=n, n_cols=n)
        dww = C * (x.T.dot(diag).dot(x)) + CArray.eye(d, d)  # matrix d*d
        dbw = C * ((z * x).sum(axis=0)).T  # column vector
        dbb = C * (z.sum(axis=None))  # scalar

        H = CArray.zeros((d + 1, d + 1))
        H[:d, :d] = dww
        H[:-1, d] = dbw
        H[d, :-1] = dbw.T
        H[-1, -1] = dbb

        return H
