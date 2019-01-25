"""
.. module:: CAttackPoisoning
   :synopsis: TODO

    @author: Battista Biggio

"""

from secml.adv.attacks.poisoning import CAttackPoisoning
from secml.array import CArray


class CAttackPoisoningSVM(CAttackPoisoning):
    """Class providing a common interface to CSolver classes."""

    class_type = 'kkt-svm'

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
                 solver_type=None,
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

    def __clear(self):
        pass

    def _init_solver(self):
        """Overrides _init_solver to additionally reset the SV indices."""
        super(CAttackPoisoningSVM, self)._init_solver()

        # reset stored indices of SVs
        self._sv_idx = None

    ###########################################################################
    #                  OBJECTIVE FUNCTION & GRAD COMPUTATION
    ###########################################################################

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
        svm, tr = self._update_poisoned_clf()

        return self._alpha_c()

    ###########################################################################
    #                            GRAD COMPUTATION
    ###########################################################################

    def _s(self, tol=1e-6):
        """Indices of margin support vectors."""
        s = self._poisoned_clf.alpha.find(
            (abs(self._poisoned_clf.alpha) >= tol) *
            (abs(self._poisoned_clf.alpha) <= self._poisoned_clf.C - tol))
        return CArray(s)

    def _ys(self):
        ys = self._poisoned_clf.alpha.sign()
        ys = CArray(ys[self._s()])
        return ys

    def _xs(self):

        s = self._s()

        if s.size == 0:
            return None

        xs = self._poisoned_clf.sv[s, :].atleast_2d()
        return xs, s

    def _alpha_c(self):
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

        # from prlib.figure import CFigure
        # fig = CFigure()
        # sp = fig.subplot()
        # sp.plot(self._poisoned_clf.alpha)
        # fig.show()

        k = self._poisoned_clf.sv_idx.find(self._poisoned_clf.sv_idx == idx)
        if len(k) == 1:  # if not empty
            return self._poisoned_clf.alpha[k]
        return 0

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

        alpha_c = self._alpha_c()

        if abs(alpha_c) == 0: # < svm.C:  # this include alpha_c == 0
            # self.logger.debug("Warning: xc is not an error vector.")
            return grad

        # take only validation points with non-null loss
        xk = self._ts.X[abs(loss_grad) > 0, :].atleast_2d()
        grad_loss_fk = CArray(loss_grad[abs(loss_grad) > 0]).atleast_2d()

        # handle normalizer, if present
        xk = xk if svm.preprocess is None else svm.preprocess.normalize(xk)
        xc = xc if svm.preprocess is None else svm.preprocess.normalize(xc)

        # gt is the gradient in feature space
        # this gradient component is the only one if margin SV set is empty
        dKkc = svm.kernel.gradient(xk, xc)
        gt = alpha_c * grad_loss_fk.dot(dKkc).ravel()

        xs, sv_idx = self._xs()  # these points are already normalized

        if xs is None:
            self.logger.debug("Warning: xs is empty "
                              "(all points are error vectors).")
         #   print "gt ", gt.norm()
            return gt if svm.preprocess is None else \
                svm.preprocess.gradient(xc0, gt)

        s = xs.shape[0]
        k = grad_loss_fk.size

        Kks_ext = CArray.ones(shape=(k, s + 1))
        Kks_ext[:, :s] = svm.kernel.k(xk, xs)
        grad_loss_params = -grad_loss_fk.dot(Kks_ext).T

        H = CArray.ones(shape=(s + 1, s + 1))
        H[:s, :s] = svm.kernel.k(xs)
        H[-1, -1] = 0
        H += 1e-9 * CArray.eye(s + 1)

        G = CArray.zeros(shape=(gt.size, s + 1))
        G[:, :s] = svm.kernel.gradient(xs, xc).T

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
        #v = - self._compute_grad_solve(G, H, grad_loss_params, sym_pos=False)

        # solve using inverse/pseudoinverse of H
        v = - self._compute_grad_inv(G, H, grad_loss_params)

        gt += v * alpha_c

        # propagating gradient back to input space
        return gt if svm.preprocess is None else \
            svm.preprocess.gradient(xc0, gt)
