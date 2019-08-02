"""
.. module:: CAttackEvasionPGD
   :synopsis: Evasion attack using Projected Gradient Descent.

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml import _NoValue
from secml.adv.attacks import CAttack
from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.optim.function import CFunction
from secml.optim.constraints import CConstraint
from secml.optim.optimizers import COptimizer


class CAttackEvasionPGD(CAttackEvasionPGDLS):
    """Evasion attacks using Projected Gradient Descent.

    It requires classifier, surrogate_classifier, and surrogate_data.
    Note that surrogate_classifier is assumed to be trained (before
    passing it to this class) on surrogate_data.

    Parameters
    ----------
    discrete: True/False (default: false).
        If True, input space is considered discrete (integer-valued),
        otherwise continuous.
    attack_classes : 'all' or CArray, optional
        List of classes that can be manipulated by the attacker or
         'all' (default) if all classes can be manipulated.
    y_target : int or None, optional
        If None an indiscriminate attack will be performed, else a
        targeted attack to have the samples misclassified as
        belonging to the y_target class.

    TODO: complete list of parameters

    Attributes
    ----------
    class_type : 'e-pgd-ls'

    """
    __class_type = 'e-pgd'

    def __init__(self, classifier,
                 surrogate_classifier,
                 surrogate_data=None,
                 distance='l1',
                 dmax=0,
                 lb=0,
                 ub=1,
                 discrete=_NoValue,
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

        # pgd solver does not accepts parameter `discrete`
        if discrete is not _NoValue:
            raise ValueError("`pgd` solver does not work in discrete space.")

        CAttack.__init__(self, classifier=classifier,
                         surrogate_classifier=surrogate_classifier,
                         surrogate_data=surrogate_data,
                         distance=distance,
                         dmax=dmax,
                         lb=lb,
                         ub=ub,
                         discrete=False,
                         y_target=y_target,
                         attack_classes=attack_classes,
                         solver_type='pgd',
                         solver_params=solver_params)

    # FIXME: THIS OVERRIDE IS REDUNDANT.
    #  `discrete` must not be passed by default
    def _init_solver(self):
        """Create solver instance."""

        if self._solver_clf is None or self.distance is None \
                or self.discrete is None:
            raise ValueError('Solver not set properly!')

        # map attributes to fun, constr, box
        fun = CFunction(fun=self._objective_function,
                        gradient=self._objective_function_gradient,
                        n_dim=self.n_dim)

        constr = CConstraint.create(self._distance)
        constr.center = self._x0
        constr.radius = self.dmax

        # only feature increments or decrements are allowed
        lb = self._x0.todense() if self.lb == 'x0' else self.lb
        ub = self._x0.todense() if self.ub == 'x0' else self.ub

        bounds = CConstraint.create('box', lb=lb, ub=ub)

        self._solver = COptimizer.create(
            self._solver_type,
            fun=fun, constr=constr,
            bounds=bounds,
            **self._solver_params)

        # TODO: fix this verbose level propagation
        self._solver.verbose = self.verbose
