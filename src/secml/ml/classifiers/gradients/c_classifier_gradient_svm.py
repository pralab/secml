from secml.array import CArray
from secml.ml.classifiers.loss import CLossHinge
from secml.ml.classifiers.gradients import CClassifierGradient


class CClassifierGradientSVM(CClassifierGradient):
    class_type = 'svm'

    def __init__(self):
        self._loss = CLossHinge()

    def hessian(self, clf):
        """
        Compute hessian of the loss w.r.t. the classifier parameters
        """
        svm = clf

        xs, sv_idx = clf.xs()  # these points are already normalized

        s = xs.shape[0]

        H = CArray.ones(shape=(s + 1, s + 1))
        H[:s, :s] = svm.kernel.k(xs)
        H[-1, -1] = 0

        return H

    def fd_params(self, clf, xk):
        """
        Derivative of the discriminant function w.r.t. the classifier
        parameters

        Parameters
        ----------
        xk : CArray
            features of a dataset
        """
        xs, sv_idx = clf.xs()  # these points are already normalized

        if xs is None:
            self.logger.debug("Warning: xs is empty "
                              "(all points are error vectors).")
            return None

        xk = xk if clf.preprocess is None else clf.preprocess.normalize(xk)

        s = xs.shape[0]  # margin support vector
        k = xk.shape[0]

        Ksk_ext = CArray.ones(shape=(s + 1, k))
        Ksk_ext[:s, :] = clf.kernel.k(xs, xk)
        return Ksk_ext  # (s + 1) * k

    def Kd_xc(self, clf, alpha_c, xc, xk):
        """
        Derivative of the kernel w.r.t. a training sample xc

        Parameters
        ----------
        xk : CArray
            features of a validation set
        xc:  CArray
            features of the training point w.r.t. the derivative has to be
            computed
        alpha_c:  integer
            alpha value of the of the training point w.r.t. the derivative has
            to be
            computed
        """
        # handle normalizer, if present
        xc = xc if clf.preprocess is None else clf.preprocess.normalize(xc)
        xk = xk if clf.preprocess is None else clf.preprocess.normalize(xk)

        dKkc = alpha_c * clf.kernel.gradient(xk, xc)
        return dKkc.T  # d * k

    def L_tot_d_params(self, clf, x, y, loss):
        """
        Derivative of the classifier classifier loss function (regularizer
        included) w.r.t. the classifier parameters

        dL / d_params = dL / df * df / d_params + dReg / d_params

        Parameters
        ----------
        x : CArray
            features of the dataset on which the loss is computed
        y :  CArray
            features of the training samples
        loss: None (default) or CLoss
            If the loss is equal to None (default) the classifier loss is used
            to compute the derivative.
        """

        if loss is None:
            loss = self._loss

        # compute the loss derivative w.r.t. alpha
        fd_params = self.fd_params(clf, x)  # (s + 1) * n_samples
        scores = clf.decision_function(x)
        dL_s = loss.dloss(y, score=scores).atleast_2d()
        dL_params = dL_s * fd_params  # (s + 1) * n_samples

        # compute the regularizer derivative w.r.t alpha
        xs, margin_sv_idx = clf.xs()
        K = clf.kernel.k(xs, xs)
        d_reg = 2 * K.dot(clf.alpha[margin_sv_idx].T)  # s * 1

        s = margin_sv_idx.size
        grad = clf.C * dL_params
        grad[:s, :] += d_reg

        return grad  # (s +1) * n_samples

    def fd_x(self, clf, x):
        """
        Derivative of the discriminant function w.r.t. a test sample
        """
        return NotImplementedError()
