"""
.. module:: CAttackPoisoning
   :synopsis: TODO

    @author: Battista Biggio

"""
from secml.adv.attacks.poisoning import CAttackPoisoning
from secml.array import CArray


class CAttackPoisoningSVM(CAttackPoisoning):
    __class_type = 'kkt-svm'

    def __init__(self, classifier,
                 training_data,
                 surrogate_classifier,
                 ts,
                 surrogate_data=None,
                 distance='l1',
                 dmax=0,
                 lb=0,
                 ub=1,
                 discrete=False,
                 y_target=None,
                 attack_classes='all',
                 solver_type='gradient-bls',
                 solver_params=None,
                 init_type='random',
                 random_seed=None):
        """
        Initialization method.

        It requires classifier, surrogate_classifier, and surrogate_data.
        Note that surrogate_classifier is assumed to be trained (before
        passing it to this class) on surrogate_data.

        TODO: complete list of parameters

        Parameters
        ------
        discrete: True/False (default: false).
                  If True, input space is considered discrete (integer-valued),
                  otherwise continuous.
        attack_classes: list of classes that can be manipulated by the attacker
                 -1 means all classes can be manipulated.

        """

        CAttackPoisoning.__init__(self, classifier=classifier,
                                  training_data=training_data,
                                  surrogate_classifier=surrogate_classifier,
                                  ts=ts,
                                  surrogate_data=surrogate_data,
                                  distance=distance,
                                  dmax=dmax,
                                  lb=lb,
                                  ub=ub,
                                  discrete=discrete,
                                  y_target=y_target,
                                  attack_classes=attack_classes,
                                  solver_type=solver_type,
                                  solver_params=solver_params,
                                  init_type=init_type,
                                  random_seed=random_seed)

        # enforce storing dual variables in SVM
        self._surrogate_classifier.store_dual_vars = True

        # indices of support vectors (at previous iteration)
        # used to check if warm_start can be used in the iterative solver
        self._sv_idx = None

    ###########################################################################
    #                           PRIVATE METHODS
    ###########################################################################

    def _init_solver(self):
        """Overrides _init_solver to additionally reset the SV indices."""
        super(CAttackPoisoningSVM, self)._init_solver()

        # reset stored indices of SVs
        self._sv_idx = None

    ###########################################################################
    #                  OBJECTIVE FUNCTION & GRAD COMPUTATION
    ###########################################################################

    def _alpha_c(self, clf):
        """
        Returns alpha value of xc, assuming xc to be appended
        as the last point in tr
        """

        # index of poisoning point within xc.
        # This will be replaced by the input parameter xc
        if self._idx is None:
            idx = 0
        else:
            idx = self._idx

        # index of the current poisoning point in the set self._xc
        # as this set is appended to the training set, idx is shifted
        idx += self._surrogate_data.num_samples

        k = clf.sv_idx.find(clf.sv_idx == idx)
        if len(k) == 1:  # if not empty
            return clf.alpha[k]
        return 0

    def alpha_xc(self, xc):
        """
        Parameters
        ----------
        xc: poisoning point

        Returns
        -------
        f_obj: values of objective function (average hinge loss) at x
        """

        # index of poisoning point within xc.
        # This will be replaced by the input parameter xc
        if self._idx is None:
            idx = 0
        else:
            idx = self._idx

        xc = CArray(xc).atleast_2d()

        n_samples = xc.shape[0]
        if n_samples > 1:
            raise TypeError("xc is not a single sample!")

        self._xc[idx, :] = xc
        # FIXME: UNUSED OUTPUT
        svm, tr = self._update_poisoned_clf()

        # FIXME: PARAMETER CLF UNFILLED
        return self._alpha_c()

    ###########################################################################
    #                            GRAD COMPUTATION
    ###########################################################################

    def _Kd_xc(self, clf, alpha_c, xc, xk):
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
        xc = xc if clf.preprocess is None else clf.preprocess.transform(xc)
        xk = xk if clf.preprocess is None else clf.preprocess.transform(xk)

        dKkc = alpha_c * clf.kernel.gradient(xk, xc)
        return dKkc.T  # d * k

    # FIXME: SIGNATURE DOES NOT MATCH WITH PARENT
    def _gradient_fk_xc(self, xc, yc, clf, loss_grad, tr):
        """
        Derivative of the classifier's discriminant function f(xk)
        computed on a set of points xk w.r.t. a single poisoning point xc

        This is a classifier-specific implementation, so we delegate its
        implementation to inherited classes.
        """

        svm = clf  # classifier is an SVM

        xc0 = xc.deepcopy()

        d = xc.size
        grad = CArray.zeros(shape=(d,))  # gradient in input space

        alpha_c = self._alpha_c(clf)

        if abs(alpha_c) == 0:  # < svm.C:  # this include alpha_c == 0
            # self.logger.debug("Warning: xc is not an error vector.")
            return grad

        # take only validation points with non-null loss
        xk = self._ts.X[abs(loss_grad) > 0, :].atleast_2d()
        grad_loss_fk = CArray(loss_grad[abs(loss_grad) > 0]).T

        # gt is the gradient in feature space
        # this gradient component is the only one if margin SV set is empty
        # gt is the derivative of the loss computed on a validation
        # set w.r.t. xc
        Kd_xc = self._Kd_xc(svm, alpha_c, xc, xk)
        gt = Kd_xc.dot(grad_loss_fk).ravel()  # gradient of the loss w.r.t. xc

        xs, sv_idx = clf.sv_margin()  # these points are already normalized

        if xs is None:
            self.logger.debug("Warning: xs is empty "
                              "(all points are error vectors).")
            return gt if svm.preprocess is None else \
                svm.preprocess.gradient(xc0, w=gt)

        s = xs.shape[0]

        # derivative of the loss computed on a validation set w.r.t. the
        # classifier params
        fd_params = svm.grad_f_params(xk)
        #grad_loss_params = fd_params.dot(-grad_loss_fk)
        grad_loss_params = fd_params.dot(grad_loss_fk)

        H = clf.hessian_tr_params()
        H += 1e-9 * CArray.eye(s + 1)

        # handle normalizer, if present
        xc = xc if clf.preprocess is None else clf.preprocess.transform(xc)
        G = CArray.zeros(shape=(gt.size, s + 1))
        G[:, :s] = svm.kernel.gradient(xs, xc).T
        G *= alpha_c

        # warm start is disabled if the set of SVs changes!
        # if self._sv_idx is None or self._sv_idx.size != sv_idx.size or \
        #         (self._sv_idx != sv_idx).any():
        #     self._warm_start = None
        # self._sv_idx = sv_idx  # store SV indices for the next iteration
        #
        # # iterative solver
        # v = - self._compute_grad_solve_iterative(
        #     G, H, grad_loss_params, tol=1e-3)

        # solve with standard linear solver
        # v = - self._compute_grad_solve(G, H, grad_loss_params, sym_pos=False)

        # solve using inverse/pseudoinverse of H
        #v = - self._compute_grad_inv(G, H, grad_loss_params)
        v = self._compute_grad_inv(G, H, grad_loss_params)

        gt += v

        # propagating gradient back to input space
        if clf.preprocess is not None:
            return clf.preprocess.gradient(xc0, w=gt)

        return gt
