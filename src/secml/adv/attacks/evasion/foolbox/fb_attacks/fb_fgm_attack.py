"""
.. module:: CFoolboxFGM
    :synopsis: Performs Foolbox FGM attack.

.. moduleauthor:: Giovanni Manca <g.manca72@studenti.unica.it>
.. moduleauthor:: Luca Demetrio <luca.demetrio@dibris.unige.it>
.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""

from foolbox.attacks.fast_gradient_method import L1FastGradientAttack, L2FastGradientAttack, LinfFastGradientAttack

from secml.adv.attacks.evasion.foolbox.c_attack_evasion_foolbox import CAttackEvasionFoolbox
from secml.adv.attacks.evasion.foolbox.losses.ce_loss import CELoss
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor

DISTANCES = ['l1', 'l2', 'linf']


class CFoolboxFGM(CELoss, CAttackEvasionFoolbox):
    """
    Fast Gradient Method Attack [Goodfellow14]_.

    Credits: https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/fast_gradient_method.py.py

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
    random_start : bool, optional
        Whether the perturbation is initialized randomly or starts at zero.

    References
    ----------
    .. [Goodfellow14] Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy
        "Explaining and Harnessing Adversarial Examples"
        https://arxiv.org/abs/1412.6572
    """
    __class_type = 'e-foolbox-fgm'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0,
                 epsilons=0.2, distance='l2',
                 random_start=True):

        if distance == 'l1':
            attack = L1FastGradientAttack
        elif distance == 'l2':
            attack = L2FastGradientAttack
        elif distance == 'linf':
            attack = LinfFastGradientAttack
        else:
            raise ValueError('Distance {} is not supported for this attack. Only {} are supported'.format(
                distance, DISTANCES
            ))

        super(CFoolboxFGM, self).__init__(classifier, y_target,
                                          lb=lb, ub=ub,
                                          fb_attack_class=attack,
                                          epsilons=epsilons,
                                          random_start=random_start)
        self._y0 = None
        self.distance = distance

    def _run(self, x, y, x_init=None):
        self._y0 = as_tensor(y)
        out, _ = super(CFoolboxFGM, self)._run(x, y, x_init)
        self._f_seq = self.objective_function(out)
        f_opt = self._f_seq[-1]
        return out, f_opt


class CFoolboxFGML1(CFoolboxFGM):
    __class_type = 'e-foolbox-fgm-l1'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0,
                 epsilons=0.2, random_start=True):
        super(CFoolboxFGML1, self).__init__(classifier, y_target,
                                            lb=lb, ub=ub,
                                            distance='l1',
                                            epsilons=epsilons,
                                            random_start=random_start)


class CFoolboxFGML2(CFoolboxFGM):
    __class_type = 'e-foolbox-fgm-l2'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0,
                 epsilon=0.2, random_start=True):
        super(CFoolboxFGML2, self).__init__(classifier, y_target,
                                            lb=lb, ub=ub,
                                            distance='l2',
                                            epsilons=epsilon,
                                            random_start=random_start)


class CFoolboxFGMLinf(CFoolboxFGM):
    __class_type = 'e-foolbox-fgm-linf'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0,
                 epsilon=0.2, random_start=True):
        super(CFoolboxFGMLinf, self).__init__(classifier, y_target,
                                              lb=lb, ub=ub,
                                              distance='linf',
                                              epsilons=epsilon,
                                              random_start=random_start)
