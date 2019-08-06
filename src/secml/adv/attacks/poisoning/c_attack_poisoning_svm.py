"""
.. module:: CAttackPoisoningSVM
   :synopsis: Poisoning attacks against Support Vector Machine

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.adv.attacks.poisoning import CAttackPoisoning
from secml.array import CArray


class CAttackPoisoningSVM(CAttackPoisoning):
    """Poisoning attacks against Support Vector Machines (SVMs).

    This is an implementation of the attack in https://arxiv.org/pdf/1206.6389:
     - B. Biggio, B. Nelson, and P. Laskov. Poisoning attacks against
       support vector machines. In J. Langford and J. Pineau, editors,
       29th Int'l Conf. on Machine Learning, pages 1807-1814. Omnipress, 2012.

    where the gradient is computed as described in Eq. (10) in
    https://www.usenix.org/conference/usenixsecurity19/presentation/demontis:
     - A. Demontis, M. Melis, M. Pintor, M. Jagielski, B. Biggio, A. Oprea,
       C. Nita-Rotaru, and F. Roli. Why do adversarial attacks transfer?
       Explaining transferability of evasion and poisoning attacks.
       In 28th USENIX Security Symposium. USENIX Association, 2019.

    For more details on poisoning attacks, see also:
     - https://arxiv.org/abs/1804.00308, IEEE Symp. SP 2018
     - https://arxiv.org/abs/1712.03141, Patt. Rec. 2018
     - https://arxiv.org/abs/1708.08689, AISec 2017
     - https://arxiv.org/abs/1804.07933, ICML 2015


    Parameters
    ----------
    classifier : CClassifierSVM
        Target classifier. If linear, requires `store_dual_vars = True`.
    training_data : CDataset
        Dataset on which the the classifier has been trained on.
    surrogate_classifier : CClassifier
        Surrogate classifier, assumed to be already trained.
    val : CDataset
        Validation set.
    surrogate_data : CDataset or None, optional
        Dataset on which the the surrogate classifier has been trained on.
        Is only required if the classifier is nonlinear.
    distance : {'l1' or 'l2'}, optional
        Norm to use for computing the distance of the adversarial example
        from the original sample. Default 'l2'.
    dmax : scalar, optional
        Maximum value of the perturbation. Default 1.
    lb, ub : int or CArray, optional
        Lower/Upper bounds. If int, the same bound will be applied to all
        the features. If CArray, a different bound can be specified for each
        feature. Default `lb = 0`, `ub = 1`.
    y_target : int or None, optional
        If None an error-generic attack will be performed, else a
        error-specific attack to have the samples misclassified as
        belonging to the `y_target` class.
    attack_classes : 'all' or CArray, optional
        Array with the classes that can be manipulated by the attacker or
         'all' (default) if all classes can be manipulated.
    solver_type : str or None, optional
        Identifier of the solver to be used. Default 'pgd-ls'.
    solver_params : dict or None, optional
        Parameters for the solver. Default None, meaning that default
        parameters will be used.
    init_type : {'random', 'loss_based'}, optional
        Strategy used to chose the initial random samples. Default 'random'.
    random_seed : int or None, optional
        If int, random_state is the seed used by the random number generator.
        If None, no fixed seed will be set.

    """
    __class_type = 'p-svm'

    def __init__(self, classifier,
                 training_data,
                 surrogate_classifier,
                 val,
                 surrogate_data=None,
                 distance='l1',
                 dmax=0,
                 lb=0,
                 ub=1,
                 discrete=False,
                 y_target=None,
                 attack_classes='all',
                 solver_type='pgd-ls',
                 solver_params=None,
                 init_type='random',
                 random_seed=None):

        CAttackPoisoning.__init__(self, classifier=classifier,
                                  training_data=training_data,
                                  surrogate_classifier=surrogate_classifier,
                                  val=val,
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
        if not self._surrogate_classifier.store_dual_vars and \
                self._surrogate_classifier.is_linear():
            raise ValueError(
                "please retrain the classifier with `store_dual_vars=True`")

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
        self._update_poisoned_clf()

        # PARAMETER CLF UNFILLED
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

    def _gradient_fk_xc(self, xc, yc, clf, loss_grad, tr, k=None):
        """
        Derivative of the classifier's discriminant function f(xk)
        computed on a set of points xk w.r.t. a single poisoning point xc
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
        xk = self._val.X[abs(loss_grad) > 0, :].atleast_2d()
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
        # grad_loss_params = fd_params.dot(-grad_loss_fk)
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

        # solve using inverse/pseudo-inverse of H
        # v = - self._compute_grad_inv(G, H, grad_loss_params)
        v = self._compute_grad_inv(G, H, grad_loss_params)

        gt += v

        # propagating gradient back to input space
        if clf.preprocess is not None:
            return clf.preprocess.gradient(xc0, w=gt)

        return gt
