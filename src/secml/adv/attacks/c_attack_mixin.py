"""
.. module:: CAttack
   :synopsis: Interface class for evasion and poisoning attacks.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""

from secml.adv.attacks import CAttack
from secml.ml.classifiers import CClassifier


class CAttackMixin(CAttack):
    """Interface class for evasion and poisoning attacks.

    Parameters
    ----------
    classifier : CClassifier
        Target classifier (trained).
    distance : {'l1' or 'l2'}, optional
        Norm to use for computing the distance of the adversarial example
        from the original sample. Default 'l2'.
    dmax : scalar, optional
        Maximum value of the perturbation. Default 1.
    lb, ub : int or CArray, optional
        Lower/Upper bounds. If int, the same bound will be applied to all
        the features. If CArray, a different bound can be specified for each
        feature. Default `lb = 0`, `ub = 1`.
    attack_classes : 'all' or CArray, optional
        Array with the classes that can be manipulated by the attacker or
         'all' (default) if all classes can be manipulated.
    solver_type : str or None, optional
        Identifier of the solver to be used.
    solver_params : dict or None, optional
        Parameters for the solver. Default None, meaning that default
        parameters will be used.

    """

    def __init__(self, classifier,
                 distance=None,
                 dmax=None,
                 lb=None,
                 ub=None,
                 solver_type=None,
                 solver_params=None):

        super(CAttackMixin, self).__init__(classifier)

        # INTERNAL
        # init attributes to None (re-defined through setters below)
        self._solver = None

        # now we populate solver parameters (via dedicated setters)
        self.dmax = dmax
        self.lb = lb
        self.ub = ub
        self.distance = distance
        self.solver_type = solver_type
        self.solver_params = solver_params

    ###########################################################################
    #                           READ-ONLY ATTRIBUTES
    ###########################################################################

    @property
    def f_eval(self):
        """Returns the number of function evaluations made during the attack.

        Warnings
        --------
        Due to a known issue, if more then one sample is passed to ``.run()``,
        this property will only return the data relative to the last
        optimized one. This behavior will change in a future version.

        """
        return self._solver.f_eval

    @property
    def grad_eval(self):
        """Returns the number of function evaluations made during the attack.

        Warnings
        --------
        Due to a known issue, if more then one sample is passed to ``.run()``,
        this property will only return the data relative to the last
        optimized one. This behavior will change in a future version.

        """
        return self._solver.grad_eval

    ###########################################################################
    #                           READ-WRITE ATTRIBUTES
    ###########################################################################

    @property
    def dmax(self):
        """Returns dmax"""
        return self._dmax

    @dmax.setter
    def dmax(self, value):
        """Sets dmax."""
        if value is None:
            self._dmax = None
            return
        self._dmax = float(value)

    @property
    def lb(self):
        """Returns lb"""
        return self._lb

    @lb.setter
    def lb(self, value):
        """Sets lb"""
        self._lb = value

    @property
    def ub(self):
        """Returns ub"""
        return self._ub

    @ub.setter
    def ub(self, value):
        """Sets ub"""
        self._ub = value

    @property
    def distance(self):
        """todo"""
        return self._distance

    @distance.setter
    def distance(self, value):
        """todo"""
        if value is None:
            self._distance = None
            return

        # check if l1 or l2
        self._distance = str(value)

    @property
    def solver_type(self):
        return self._solver_type

    @solver_type.setter
    def solver_type(self, value):
        self._solver_type = None if value is None else str(value)

    @property
    def solver_params(self):
        return self._solver_params

    @solver_params.setter
    def solver_params(self, value):
        self._solver_params = {} if value is None else dict(value)

    ###########################################################################
    #                              METHODS
    ###########################################################################

    def _solution_from_solver(self):
        """Retrieve solution from solver and set internal class parameters."""
        # retrieve sequence of evasion points, and final point
        self._x_seq = self._solver.x_seq
        self._x_opt = self._solver.x_opt

        # retrieve f_obj values on x_seq and x_opt (from solver)
        self._f_seq = self._solver.f_seq
        self._f_opt = self._solver.f_opt
