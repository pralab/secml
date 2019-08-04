"""
.. module:: CAttackPoisoningLogisticRegression
   :synopsis: Poisoning attacks against logistic regression

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml.adv.attacks.poisoning import CAttackPoisoning
from secml.array import CArray
from secml.ml.classifiers.clf_utils import convert_binary_labels


class CAttackPoisoningLogisticRegression(CAttackPoisoning):
    __class_type = 'p-logistic'

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

    def _s(self, x, w, b):
        """
        Classifier score

        :param x:
        :param w:
        :param b:
        :return: score: column vector
        """
        return x.dot(w) + b

    def _sigm(self, y, s):
        """
        Sigmoid function

        :param y:
        :param s:
        :return:
        """
        y = CArray(y)
        s = CArray(s)
        return 1.0 / (1.0 + (-y * s).exp())

    def _gradient_fk_xc(self, xc, yc, clf, loss_grad, tr):
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
