"""
.. module:: CAttackEvasionPGD
   :synopsis: Evasion attack using Projected Gradient Descent.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml import _NoValue
from secml.adv.attacks import CAttack
from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.optim.function import CFunction
from secml.optim.constraints import CConstraint
from secml.optim.optimizers import COptimizer


class CAttackEvasionPGD(CAttackEvasionPGDLS):
    """Evasion attacks using Projected Gradient Descent.

    This class implements the maximum-confidence evasion attacks proposed in:
     - https://arxiv.org/abs/1708.06939, ICCV W. ViPAR, 2017.

    This is the multi-class extension of our original work in:
     - https://arxiv.org/abs/1708.06131, ECML 2013,
       implemented using a standard projected gradient solver.

    It can also be used on sparse, high-dimensional feature spaces, using an
    L1 constraint on the manipulation of samples to preserve sparsity,
    as we did for crafting adversarial Android malware in:
     - https://arxiv.org/abs/1704.08996, IEEE TDSC 2017.

    For more on evasion attacks, see also:
     - https://arxiv.org/abs/1809.02861, USENIX Sec. 2019
     - https://arxiv.org/abs/1712.03141, Patt. Rec. 2018

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
