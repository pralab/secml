"""
.. module:: CFoolboxDeepfool
    :synopsis: Performs Foolbox Deepfool attack in L2 and Linf norms.

.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""

from foolbox.attacks.deepfool import L2DeepFoolAttack, LinfDeepFoolAttack

from secml.adv.attacks.evasion.foolbox.c_attack_evasion_foolbox import \
    CAttackEvasionFoolbox
from secml.adv.attacks.evasion.foolbox.losses.deepfool_loss import \
    DeepfoolLoss
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor

CELOSS = 'crossentropy'
LOGITLOSS = 'logits'
DISTANCES = ['l2', 'linf']


class CFoolboxDeepfool(DeepfoolLoss, CAttackEvasionFoolbox):
    """
    Deepfool Attack [#moosavidezfooli15]_.

    Credits: https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/deepfool.py

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
    steps : int, optional
        Maximum number of steps to perform.
    candidates : int, optional
        Limit on the number of the most likely classes that
        should be considered. A small value is usually sufficient and
        much faster.
    overshoot : float, optional
        How much to overshoot the boundary.
    loss : str, optional
        Loss function to use inside the update function. Supported
        losses are 'crossentropy' and 'logits'.

    References
    ----------
    .. [#moosavidezfooli15] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
        "DeepFool: a simple and accurate method to fool deep neural
        networks", https://arxiv.org/abs/1511.04599
    """
    __class_type = 'e-foolbox-deepfool'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0,
                 epsilons=0.2, distance='l2', steps=50,
                 candidates=10, overshoot=0.02, loss="logits"):
        if y_target != None:
            raise ValueError(
                "Unsupported criterion. Deepfool only "
                "supports the untargeted version.")
        if distance == 'l2':
            attack = L2DeepFoolAttack
        elif distance == 'linf':
            attack = LinfDeepFoolAttack
        else:
            raise ValueError(
                'Distance {} is not supported for this attack. Only {} '
                'are supported'.format(
                    distance, DISTANCES
                ))
        super(CFoolboxDeepfool, self).__init__(
            classifier, y_target,
            lb=lb, ub=ub,
            fb_attack_class=attack,
            epsilons=epsilons, steps=steps,
            candidates=candidates,
            overshoot=overshoot,
            loss=loss)
        self._x0 = None
        self._y0 = None
        self.distance = distance
        self.loss = loss
        self.candidates = candidates

    def _run(self, x, y, x_init=None):
        self._x0 = as_tensor(x)
        self._y0 = as_tensor(y)
        out, _ = super(CFoolboxDeepfool, self)._run(x, y, x_init)
        self._f_seq = self.objective_function(self.x_seq)
        f_opt = self.objective_function(out)
        return out, f_opt


class CFoolboxDeepfoolL2(CFoolboxDeepfool):
    __class_type = 'e-foolbox-deepfool-l2'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0,
                 epsilons=0.2, steps=50, candidates=10, overshoot=0.02,
                 loss="logits"):
        super(CFoolboxDeepfoolL2, self).__init__(
            classifier, y_target,
            lb=lb, ub=ub,
            distance='l2',
            epsilons=epsilons, steps=steps,
            candidates=candidates,
            overshoot=overshoot,
            loss=loss)


class CFoolboxDeepfoolLinf(CFoolboxDeepfool):
    __class_type = 'e-foolbox-deepfool-linf'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0,
                 epsilons=0.2, steps=50, candidates=10, overshoot=0.02,
                 loss="logits"):
        super(CFoolboxDeepfoolLinf, self).__init__(
            classifier, y_target,
            lb=lb, ub=ub,
            distance='linf',
            epsilons=epsilons, steps=steps,
            candidates=candidates,
            overshoot=overshoot,
            loss=loss)
