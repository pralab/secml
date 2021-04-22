"""
.. module:: CClassifierSecSVM
   :synopsis: Secure Support Vector Machine (SVM) classifier

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray
from secml.core.constants import inf
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.clf_utils import convert_binary_labels


class CClassifierSecSVM(CClassifierSVM):
    """Secure Support Vector Machine (Sec-SVM) classifier.

    This implements the secure classifier from:

     Demontis et al. "Yes, machine learning can be more secure! a case study
     on android malware detection." IEEE TDSC 2017. https://arxiv.org/abs/1704.08996

    Parameters
    ----------
    ub : scalar or None, optional
        Upper bound of the weights. If None (default), no bound is applied.
    idx_ub : CArray or None, optional
        If CArray, the upper bound is only applied to the weights indicized
        by `idx_ub`. If None (default), the bound is applied to all weights.
    lb : scalar or None, optional
        Lower bound of the weights. If None (default), no bound is applied.
    idx_lb : CArray or None, optional
        If CArray, the lower bound is only applied to the weights indicized
        by `idx_ub`. If None (default), the bound is applied to all weights.
    eta : scalar, optional
        Step of the gradient descent. Default 0.5.
    max_it : int, optional
        Maximum number of iterations of the gradient descent. Default 1e4.
    eps : scalar, optional
        Tolerance of the stop criterion of the gradient descent. Default 1e-4.
    *args, **kwargs
        Other paramters from :class:`CClassifierSVM`.

    Attributes
    ----------
    class_type : 'sec-svm'

    """
    __class_type = 'sec-svm'

    def __init__(self, ub=inf, idx_ub=None, lb=-inf, idx_lb=None,
                 eta=0.5, max_it=1e4, eps=1e-4, *args, **kwargs):

        # Calling standard CClassifierSVM constructor
        super(self.__class__, self).__init__(*args, **kwargs)

        # Training parameters
        self.eta = eta
        self.max_it = max_it
        self.eps = eps

        # Gradient bounds
        if (CArray(ub) < CArray(lb)).any():
            raise ValueError("Upper bounds should be higher then lower bounds")

        self._ub = ub
        self._idx_ub = idx_ub if idx_ub is not None \
            else slice(None, None, None)
        self._lb = lb
        self._idx_lb = idx_lb if idx_lb is not None \
            else slice(None, None, None)

    @property
    def ub(self):
        """Return value of weight upper bound"""
        return self._ub

    @property
    def lb(self):
        """Return value of weight lower bound"""
        return self._lb

    @property
    def w(self):
        """Return the vector of feature weights (only if ckernel is None)"""
        return self._w

    @property
    def b(self):
        """Return the SVM bias (b term in the decision function)"""
        return self._b

    @property
    def max_it(self):
        """Maximum number of iteration for the training."""
        return self._max_it

    @max_it.setter
    def max_it(self, value):
        """Maximum number of iteration for the training."""
        self._max_it = int(value)

    @property
    def eta(self):
        """Eta parameter for the training gradient."""
        return self._eta

    @eta.setter
    def eta(self, value):
        """Eta parameter for the training gradient."""
        self._eta = float(value)

    @property
    def eps(self):
        """Precision of the stop condition for training."""
        return self._eta

    @eps.setter
    def eps(self, value):
        """Precision of the stop condition for training."""
        self._eta = float(value)

    def hinge_loss(self, x, y):
        """Compute the loss term for each point in dataset."""
        score = self.decision_function(x, y=1)
        loss = 1.0 - y * score
        loss[loss < 0] = 0.0
        return loss

    def C_hinge_loss(self, x, y):
        """Compute the loss term for each point in dataset multiplied by C.

        If class_weight == 'balanced', it multiplies C to the inverse
        prob of theclasses.

        """
        loss = self.C * self.hinge_loss(x, y)

        if self.class_weight == 'balanced':
            loss[y == -1] = self.weight[0] * loss[y == -1]
            loss[y == 1] = self.weight[1] * loss[y == 1]

        return loss

    def gradient_w_b(self, x, y):
        """
        Compute the gradient dloss/dw, where loss is \sum_i max(0, 1-y_i*f(x_i))
        """
        loss = self.hinge_loss(x, y)  # loss(y,f(x))

        idx_err_vect = loss > 0

        grad_b = -self.C * y[idx_err_vect].sum()

        grad_loss = CArray.zeros(x.shape[1])
        if (idx_err_vect * (y < 0)).any():
            grad_loss += x[idx_err_vect * (y <= 0), :].sum(
                axis=0, keepdims=False)
        if (idx_err_vect * (y > 0)).any():
            grad_loss -= x[idx_err_vect * (y > 0), :].sum(
                axis=0, keepdims=False)

        grad_w = self.w + self.C * grad_loss

        return grad_w, grad_b

    def objective(self, x, y):
        """Objective function."""
        return 0.5 * self.w.dot(self.w.T) + self.C_hinge_loss(x, y).sum()

    def _fit(self, x, y):
        """Trains the One-Vs-All SVM classifier.

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features).
        y : CArray
            Array of shape (n_samples,) containing the class
            labels (2-classes only).

        Returns
        -------
        CClassifierSecSVM
            Trained classifier.

        """
        if self.n_classes != 2:
            raise ValueError(
                "Trying to learn an SVM on more/less than two classes.")

        y = convert_binary_labels(y)

        if self.class_weight == 'balanced':
            n_pos = y[y == 1].shape[0]
            n_neg = y[y == -1].shape[0]
            self.weight = CArray.zeros(2)
            self.weight[0] = 1.0 * n_pos / (n_pos + n_neg)
            self.weight[1] = 1.0 * n_neg / (n_pos + n_neg)

        self._w = CArray.zeros(x.shape[1])
        self._b = CArray(0.0)

        obj = self.objective(x, y)
        obj_new = obj

        for i in range(self.max_it):

            # pick a random sample subset
            idx = CArray.randsample(
                CArray.arange(x.shape[0], dtype=int), x.shape[0],
                random_state=i)

            # compute subgradients
            grad_w, grad_b = self.gradient_w_b(x[idx, :], y[idx])

            for p in range(0, 71, 10):

                step = (self.eta ** p) * 2 ** (-0.01 * i) / (x.shape[0] ** 0.5)

                self._w -= step * grad_w
                self._b -= step * grad_b

                # Applying UPPER bound
                d_ub = self.w[self._idx_ub]
                d_ub[d_ub > self._ub] = self._ub
                self.w[self._idx_ub] = d_ub

                # Applying LOWER bound
                d_lb = self.w[self._idx_lb]
                d_lb[d_lb < self._lb] = self._lb
                self.w[self._idx_lb] = d_lb

                obj_new = self.objective(x, y)

                if obj_new < obj:
                    break

            if abs(obj_new - obj) < self.eps:
                self.logger.info("i {:}: {:}".format(i, obj_new))
                # Sparse weights if input is sparse (like in CClassifierSVM)
                self._w = self.w.tosparse() if x.issparse else self.w
                return

            obj = obj_new

            if i % 10 == 0:
                loss = self.hinge_loss(x, y).sum()
                self.logger.info(
                    "i {:}: {:.4f}, L {:.4f}".format(i, obj, loss))
            # Sparse weights if input is sparse (like in CClassifierSVM)
            self._w = self.w.tosparse() if x.issparse else self.w
