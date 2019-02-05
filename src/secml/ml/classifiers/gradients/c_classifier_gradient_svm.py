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

        Ksk_ext = CArray.ones(shape=(s + 1, k))
        Ksk_ext[:s, :] = clf.kernel.k(xs, xk)
        return Ksk_ext  # (s + 1) * k

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
        return dKkc.T  # d * k

    def L_tot_d_params(self, x, y, clf):
        """
        Derivative of the classifier classifier loss function (regularizer
        included) w.r.t. the classifier parameters

        dL / d_params = dL / df * df / d_params + dReg / d_params

        x : CArray
            features of the training sample
        y :  CArray
            features of the training samples
        """
        # compute the loss derivative w.r.t. alpha
        fd_params = self.fd_params(x, clf)  # (s + 1) * n_samples
        scores = clf.decision_function(x)
        dL_s = self._loss.dloss(y, score=scores).atleast_2d()
        dL_params = dL_s * fd_params  # (s + 1) * n_samples

        # compute the regularizer derivative w.r.t alpha
        sv = clf.sv
        K = clf.kernel.k(sv, sv)
        d_reg = 2 * K.dot(clf.alpha.T)  # s * 1

        s = sv.shape[0]
        grad = clf.C * dL_params[:s, :] + d_reg

        return grad  # (s +1) * n_samples

    def _L_tot(self, x, y, clf):
        """
        Classifier total loss
        L_tot = loss computed on the training samples + regularizer
        """

        # compute the loss on the training samples
        fd_params = self.fd_params(x, clf)  # (s + 1) * n_samples
        scores = clf.decision_function(x)
        loss = self._loss.loss(y, score=scores).atleast_2d()

        # compute the value of the regularizer
        sv = clf.sv
        K = clf.kernel.k(sv, sv)
        reg = 1. / 2 * clf.alpha.atleast_2d().dot(K.dot(clf.alpha.T))

        loss = clf.C * loss + reg

        return loss

    def _params(self, clf):
        """
        Classifier parameters
        """
        return clf.alpha.append(CArray(clf.b), axis=None)

    def _change_params(self,params, clf):

        new_clf = clf.deepcopy()
        new_clf._alpha = params[:-1]
        new_clf._b = params[-1]
        return new_clf