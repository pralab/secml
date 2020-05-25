"""
.. module:: CAttackEvasionPGDExp
   :synopsis: Evasion attack using Projected Gradient Descent.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from secml import _NoValue
from secml.adv.attacks import CAttack
from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.optim.function import CFunction
from secml.optim.constraints import CConstraint
from secml.optim.optimizers import COptimizer


class CAttackEvasionPGDExp(CAttackEvasionPGDLS):
    """Evasion attacks using Projected Gradient Descent with Exponential line search.

    This class implements the maximum-confidence evasion attacks proposed in:

     - https://arxiv.org/abs/1910.00470, EURASIP JIS, 2020.
     - https://arxiv.org/abs/1708.06939, ICCV W. ViPAR, 2017.

    It is the multi-class extension of our original work in:

     - https://arxiv.org/abs/1708.06131, ECML 2013,
       implemented using a standard projected gradient solver.

    This attack uses a faster line search than PGD-LS.

    In all our attacks, we use a smart double initialization to avoid using the
    mimicry term from our ECML 2013 paper, as described in:
    - https://pralab.diee.unica.it/sites/default/files/zhang15-tcyb.pdf, IEEE TCYB, 2015

    If the attack is not successful when starting from x0,
    we initialize the optimization by projecting a point from another
    class onto the feasible domain and try again.

    Parameters
    ----------
    classifier : CClassifier
        Target classifier.
    surrogate_classifier : CClassifier
        Surrogate classifier, assumed to be already trained.
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
    solver_params : dict or None, optional
        Parameters for the solver. Default None, meaning that default
        parameters will be used.

    Attributes
    ----------
    class_type : 'e-pgd-exp'

    """
    __class_type = 'e-pgd-exp'

    def __init__(self, classifier,
                 surrogate_classifier,
                 surrogate_data=None,
                 distance='l1',
                 dmax=0,
                 lb=0,
                 ub=1,
                 discrete=False,
                 y_target=None,
                 attack_classes='all',
                 solver_params=None):

        # INTERNALS
        self._x0 = None
        self._y0 = None

        # this is an alternative init point. This could be a single point
        # (targeted evasion) or an array of multiple points, one for each
        # class (indiscriminate evasion). See _get_point_with_min_f_obj()
        self._xk = None

        super(CAttackEvasionPGDExp, self).__init__(
            classifier=classifier,
            surrogate_classifier=surrogate_classifier,
            surrogate_data=surrogate_data,
            distance=distance,
            dmax=dmax,
            lb=lb,
            ub=ub,
            discrete=discrete,
            y_target=y_target,
            attack_classes=attack_classes,
            solver_params=solver_params)

        self.solver_type = 'pgd-exp'
