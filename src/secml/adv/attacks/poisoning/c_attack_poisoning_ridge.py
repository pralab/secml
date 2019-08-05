"""
.. module:: CAttackPoisoningRidge
   :synopsis: Poisoning attacks against ridge

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.adv.attacks.poisoning import CAttackPoisoning
from secml.array import CArray
from secml.ml.classifiers.clf_utils import convert_binary_labels


class CAttackPoisoningRidge(CAttackPoisoning):
    """Poisoning attacks against ridge.

    Parameters
    ----------
    classifier : CClassifierRidge
        Target classifier.
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
    __class_type = 'p-ridge'

    def __init__(self, classifier,
                 training_data,
                 surrogate_classifier,
                 val,
                 surrogate_data=None,
                 distance='l2',
                 dmax=0,
                 lb=0,
                 ub=1,
                 discrete=False,
                 y_target=None,
                 attack_classes='all',
                 solver_type='pgd-ls',
                 solver_params=None,
                 init_type=None,
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

    ###########################################################################
    #                            GRAD COMPUTATION
    ###########################################################################

    def _g(self, d):

        return CArray.eye(d)

    # the differences with the general attack class for quadratic losses are
    # the computing of _g and the fact that here the bias is regularized
    # (only M should be the other difference)
    # FIXME: SIGNATURE DOES NOT MATCH PARENT
    def _gradient_fk_xc(self, xc, yc, clf, loss_grad, tr):
        """
        Derivative of the classifier's discriminant function f(xk)
        computed on a set of points xk w.r.t. a single poisoning point xc

        This is a classifier-specific implementation, so we delegate its
        implementation to inherited classes.
        """

        # fixme: add a paramer for this as if we are attacking a regressor
        #  we shoudn't do this.
        yc = convert_binary_labels(yc)

        xc0 = xc.deepcopy()

        # take validation points
        xk = self._val.X.atleast_2d()
        x = tr.X.atleast_2d()

        H = clf.hessian_tr_params(x)

        grad_loss_fk = CArray(loss_grad.ravel()).T  # column vector

        # handle normalizer, if present
        xc = xc if clf.preprocess is None else clf.preprocess.transform(xc)
        xc = xc.ravel().atleast_2d()
        #xk = xk if clf.preprocess is None else clf.preprocess.transform(xk)

        # gt is the gradient in feature space
        k = xk.shape[0]  # num validation samples
        d = xk.shape[1]  # num features

        M = clf.w.T.dot(
            xc)  # xc is column, w is row (this is an outer product)
        M += (clf.w.dot(xc.T) + clf.b - yc) * CArray.eye(d)
        db_xc = clf.w.T
        G = M.append(db_xc, axis=1)

        # add diagonal noise to the matrix that we are gong to invert
        H += 1e-9 * (CArray.eye(d + 1))

        # # compute the derivatives of the classifier discriminant function
        fd_params = self.classifier.grad_f_params(xk)
        grad_loss_params = fd_params.dot(grad_loss_fk)

        # import time
        # start = time.time()

        # gt is the gradient in feature space
        gt = self._compute_grad_inv(G, H, grad_loss_params)
        # gt = self._compute_grad_solve(G, H, grad_loss_params)
        # gt = self._compute_grad_solve_iterative(G, H, grad_loss_params) #*

        # end = time.time()
        # print "time: ", end - start

        # propagating gradient back to input space
        if clf.preprocess is not None:
            return clf.preprocess.gradient(xc0, w=gt)

        return gt

