"""
.. module:: CAttackPoisoningRidge
   :synopsis: Poisoning attacks against ridge

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.adv.attacks.poisoning import CAttackPoisoning
from secml.array import CArray
from secml.ml.classifiers.clf_utils import convert_binary_labels


class CAttackPoisoningRidge(CAttackPoisoning):
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
        """
        Initialization method.

        It requires classifier, surrogate_classifier, and surrogate_data.
        Note that surrogate_classifier is assumed to be trained (before
        passing it to this class) on surrogate_data.

        TODO: complete list of parameters

        Parameters
        ----------
        discrete: True/False (default: false).
                  If True, input space is considered discrete (integer-valued),
                  otherwise continuous.
        attack_classes: list of classes that can be manipulated by the attacker
                 -1 means all classes can be manipulated.

        """

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
        """
        :param d: number of features
        :return:
        """
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

