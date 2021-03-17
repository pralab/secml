"""
.. module:: CFoolboxBasicIterative
    :synopsis: Performs Foolbox Basic Iterative attack.

.. moduleauthor:: Giovanni Manca <g.manca72@studenti.unica.it>
.. moduleauthor:: Luca Demetrio <luca.demetrio@dibris.unige.it>
.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""

from foolbox.attacks.basic_iterative_method import L1BasicIterativeAttack, L2BasicIterativeAttack, \
    LinfBasicIterativeAttack

from secml.adv.attacks.evasion.foolbox.c_attack_evasion_foolbox import CAttackEvasionFoolbox
from secml.adv.attacks.evasion.foolbox.losses.ce_loss import CELoss
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor

DISTANCES = ['l1', 'l2', 'linf']


class CFoolboxBasicIterative(CELoss, CAttackEvasionFoolbox):
    """
    Basic Iterative Method Attack [Kurakin16]_.

    Credits: https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/basic_iterative_method.py

    Parameters
    ----------
    classifier : CClassifier
        Trained secml classifier.
    y_target : int or None, optional
        If None an indiscriminate attack will be performed, else a
        targeted attack to have the samples misclassified as
        belonging to the y_target class.
    lb : float or None, optional
        Lower bound of the model's input space.
    ub : float or None, optional
         Upper bound of the model's input space.
    epsilons : float or None, optional
        The maximum size of the perturbations, required for the
        fixed epsilon foolbox attacks.
    distance : str, optional
        Norm of the attack. One of 'l1', 'l2', 'linf'.
    rel_stepsize : float, optional
        Stepsize relative to epsilon.
    abs_stepsize : float, optional
        If given, it takes precedence over rel_stepsize.
    steps : int, optional
        Number of update steps to perform.
    random_start : bool, optional
        Whether the perturbation is initialized randomly or starts at zero.

    References
    ----------
    .. [Kurakin16] Alexey Kurakin, Ian Goodfellow, Samy Bengio
        "Adversarial examples in the physical world"
        https://arxiv.org/abs/1607.02533
    """
    __class_type = 'e-foolbox-basiciterative'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0,
                 epsilons=0.2, distance='l2',
                 rel_stepsize=0.025, abs_stepsize=None, steps=50,
                 random_start=True):

        if distance == 'l1':
            attack = L1BasicIterativeAttack
        elif distance == 'l2':
            attack = L2BasicIterativeAttack
        elif distance == 'linf':
            attack = LinfBasicIterativeAttack
        else:
            raise ValueError('Distance {} is not supported for this attack. Only {} are supported'.format(
                distance, DISTANCES
            ))

        super(CFoolboxBasicIterative, self).__init__(classifier, y_target,
                                                     lb=lb, ub=ub,
                                                     fb_attack_class=attack,
                                                     epsilons=epsilons,
                                                     rel_stepsize=rel_stepsize,
                                                     abs_stepsize=abs_stepsize,
                                                     steps=steps,
                                                     random_start=random_start)

        self._x0 = None
        self._y0 = None
        self.distance = distance

    def _run(self, x, y, x_init=None):
        self._x0 = as_tensor(x)
        self._y0 = as_tensor(y)
        out, _ = super(CFoolboxBasicIterative, self)._run(x, y, x_init)
        self._f_seq = self.objective_function(self.x_seq)
        f_opt = self.objective_function(out)
        return out, f_opt


class CFoolboxBasicIterativeL1(CFoolboxBasicIterative):
    __class_type = 'e-foolbox-basiciterative-l1'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0, epsilons=0.2,
                 rel_stepsize=0.025, abs_stepsize=None,
                 steps=50, random_start=True):
        super(CFoolboxBasicIterativeL1, self).__init__(classifier, y_target,
                                                       lb=lb, ub=ub,
                                                       distance='l1',
                                                       epsilons=epsilons,
                                                       rel_stepsize=rel_stepsize,
                                                       abs_stepsize=abs_stepsize,
                                                       steps=steps,
                                                       random_start=random_start)


class CFoolboxBasicIterativeL2(CFoolboxBasicIterative):
    __class_type = 'e-foolbox-basiciterative-l2'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0, epsilons=0.2,
                 rel_stepsize=0.025, abs_stepsize=None,
                 steps=50, random_start=True):
        super(CFoolboxBasicIterativeL2, self).__init__(classifier, y_target,
                                                       lb=lb, ub=ub,
                                                       distance='l2',
                                                       epsilons=epsilons,
                                                       rel_stepsize=rel_stepsize,
                                                       abs_stepsize=abs_stepsize,
                                                       steps=steps,
                                                       random_start=random_start)


class CFoolboxBasicIterativeLinf(CFoolboxBasicIterative):
    __class_type = 'e-foolbox-basiciterative-linf'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0, epsilons=0.2,
                 rel_stepsize=0.025, abs_stepsize=None,
                 steps=50, random_start=True):
        super(CFoolboxBasicIterativeLinf, self).__init__(classifier, y_target,
                                                         lb=lb, ub=ub,
                                                         distance='linf',
                                                         epsilons=epsilons,
                                                         rel_stepsize=rel_stepsize,
                                                         abs_stepsize=abs_stepsize,
                                                         steps=steps,
                                                         random_start=random_start)
