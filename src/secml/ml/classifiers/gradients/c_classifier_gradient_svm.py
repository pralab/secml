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

    def fd_params(self, xk, clf):
        """
        Derivative of the discriminant function w.r.t. the classifier
        parameters

        Parameters
        ----------
        xk : CArray
            features of a validation set
        """
        xs, sv_idx = clf.xs()  # these points are already normalized

        if xs is None:
            self.logger.debug("Warning: xs is empty "
                              "(all points are error vectors).")
            return None

        xk = xk if clf.preprocess is None else clf.preprocess.normalize(xk)

        s = xs.shape[0]
        k = xk.shape[0]

        Kks_ext = CArray.ones(shape=(k, s + 1))
        Kks_ext[:, :s] = clf.kernel.k(xk, xs)
        return Kks_ext

    def fd_x(self, alpha_c, xc, xk, clf):
        """
        Derivative of the discriminant function w.r.t. an input sample

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
        return dKkc.T # d * k

    def L_tot_d_params(self, x, y, clf):
        raise NotImplementedError()
