"""
.. module:: CAttackPoisoningLogisticRegression
   :synopsis: Poisoning attacks against logistic regression

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from secml.adv.attacks.poisoning import CAttackPoisoning
from secml.array import CArray
from secml.ml.classifiers.clf_utils import convert_binary_labels


class CAttackPoisoningLogisticRegression(CAttackPoisoning):
    """Poisoning attacks against logistic regression.

    This is an implementation of the attack developed in Sect. 3.3 in
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
     - https://arxiv.org/pdf/1206.6389, ICML 2012

    Parameters
    ----------
    classifier : CClassifierLogistic
        Target classifier.
    training_data : CDataset
        Dataset on which the the classifier has been trained on.
    val : CDataset
        Validation set.
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
    __class_type = 'p-logistic'

    def __init__(self, classifier,
                 training_data,
                 val,
                 distance='l1',
                 dmax=0,
                 lb=0,
                 ub=1,
                 y_target=None,
                 solver_type='pgd-ls',
                 solver_params=None,
                 init_type='random',
                 random_seed=None):

        CAttackPoisoning.__init__(self, classifier=classifier,
                                  training_data=training_data,
                                  val=val,
                                  distance=distance,
                                  dmax=dmax,
                                  lb=lb,
                                  ub=ub,
                                  y_target=y_target,
                                  solver_type=solver_type,
                                  solver_params=solver_params,
                                  init_type=init_type,
                                  random_seed=random_seed)

    ###########################################################################
    #                            GRAD COMPUTATION
    ###########################################################################

    def _s(self, x, w, b):
        """Compute classifier score."""
        return x.dot(w) + b

    def _sigm(self, y, s):
        """Compute sigmoid function."""
        y = CArray(y)
        s = CArray(s)
        return 1.0 / (1.0 + (-y * s).exp())

    def _gradient_fk_xc(self, xc, yc, clf, loss_grad, tr, k=None):
        """
        Derivative of the classifier's discriminant function f(xk)
        computed on a set of points xk w.r.t. a single poisoning point xc

        This is a classifier-specific implementation, so we delegate its
        implementation to inherited classes.

        """
        xc0 = xc.deepcopy()

        d = xc.size

        if hasattr(clf, 'C'):
            C = clf.C
        elif hasattr(clf, 'alpha'):
            C = 1.0 / clf.alpha
        else:
            raise ValueError("Error: The classifier does not have neither C "
                             "nor alpha")

        H = clf.hessian_tr_params(tr.X, tr.Y)

        # change vector dimensions to match the mathematical formulation...
        yc = convert_binary_labels(yc)
        xc = CArray(xc.ravel()).atleast_2d()  # xc is a row vector

        w = CArray(clf.w.ravel()).T  # column vector
        b = clf.b
        grad_loss_fk = CArray(loss_grad.ravel()).T  # column vector

        # validation points
        xk = self.val.X.atleast_2d()

        # handle normalizer, if present
        xc = xc if clf.preprocess is None else clf.preprocess.transform(xc)

        s_c = self._s(xc, w, b)
        sigm_c = self._sigm(yc, s_c)
        z_c = sigm_c * (1 - sigm_c)

        dbx_c = z_c * w  # column vector
        dwx_c = ((yc * (-1 + sigm_c)) * CArray.eye(d, d)) + z_c * (
            w.dot(xc))  # matrix d*d

        G = C * (dwx_c.append(dbx_c, axis=1))

        fd_params = self.classifier.grad_f_params(xk)
        grad_loss_params = fd_params.dot(grad_loss_fk)

        gt = self._compute_grad_inv(G, H, grad_loss_params)
        # gt = self._compute_grad_solve(G, H, grad_loss_params)
        # gt = self._compute_grad_solve_iterative(G, H, grad_loss_params) #*

        # propagating gradient back to input space
        if clf.preprocess is not None:
            return clf.preprocess.gradient(xc0, w=gt)

        return gt
