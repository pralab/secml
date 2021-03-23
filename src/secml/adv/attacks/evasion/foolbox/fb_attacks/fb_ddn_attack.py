"""
.. module:: CFoolboxL2DDN
    :synopsis: Performs Foolbox DDN L2 attack.


.. moduleauthor:: Giovanni Manca <g.manca72@studenti.unica.it>
.. moduleauthor:: Luca Demetrio <luca.demetrio@dibris.unige.it>
.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""

from foolbox.attacks.ddn import DDNAttack

from secml.adv.attacks.evasion.foolbox.c_attack_evasion_foolbox import CAttackEvasionFoolbox
from secml.adv.attacks.evasion.foolbox.losses.ce_loss import CELoss
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor


class CFoolboxL2DDN(CELoss, CAttackEvasionFoolbox):
    """
    Decoupling Direction and Norm Attack [#Rony18]_.

    Credits: https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/ddn.py

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
    init_epsilon : float, optional
        Initial value for the norm/epsilon ball.
    steps : int, optional
        Number of steps for the optimization.
    gamma : float, optional
        Factor by which the norm will be modified: new_norm =
        norm * (1 + or - gamma).

    References
    ----------
    .. [#Rony18] Jérôme Rony, Luiz G. Hafemann, Luiz S. Oliveira, Ismail Ben Ayed,
        Robert Sabourin, Eric Granger, "Decoupling Direction and Norm for
        Efficient Gradient-Based L2 Adversarial Attacks and Defenses",
        https://arxiv.org/abs/1811.09600
        """
    __class_type = 'e-foolbox-ddn'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0,
                 epsilons=None, init_epsilon=1.0, steps=10,
                 gamma=0.05, ):
        attack_params = {'init_epsilon': init_epsilon,
                         'gamma': gamma,
                         'steps': steps, 'epsilons': epsilons}

        super(CFoolboxL2DDN, self).__init__(classifier, y_target,
                                            lb=lb, ub=ub,
                                            fb_attack_class=DDNAttack,
                                            **attack_params)
        self._y0 = None
        self.distance = 'l2'

    def _run(self, x, y, x_init=None):
        self._y0 = as_tensor(y)
        out, _ = super(CFoolboxL2DDN, self)._run(x, y, x_init)
        self._f_seq = self.objective_function(self.x_seq)
        f_opt = self.objective_function(out)
        return out, f_opt
