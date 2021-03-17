"""
.. module:: CFoolboxEAD
    :synopsis: Performs Foolbox EAD attack.

.. moduleauthor:: Luca Demetrio <luca.demetrio@dibris.unige.it>
.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""
import math
from typing import Any, Tuple

import eagerpy as ep
from foolbox import Misclassification, TargetedMisclassification
from foolbox.attacks.base import raise_if_kwargs, get_criterion
from foolbox.attacks.ead import EADAttack, _best_other_classes, _project_shrinkage_thresholding, _apply_decision_rule

from secml.adv.attacks.evasion.foolbox.c_attack_evasion_foolbox import CAttackEvasionFoolbox
from secml.adv.attacks.evasion.foolbox.losses.ead_loss import EADLoss
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor
from secml.array import CArray
from secml.ml import CClassifier

L1 = "L1"

EN = "EN"


class CFoolboxEAD(EADLoss, CAttackEvasionFoolbox):
    """
    EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples [#Chen17]_.

    Credits: https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/ead.py

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
    binary_search_steps : int, Optional
        Number of steps used by the binary search algorithm
        for tuning c, starting from the initial_const.
    steps : int, optional
        Number of steps for the optimization.
    initial_stepsize : float, Optional
        The initial step size for the search.
    confidence : float, Optional
        Specifies how much the attacker should enter inside the target class.
    initial_const : float, Optional
        Initial constant c used during the attack.
    regularization : float, Optional
        Controls the L1 regularization.
    decision_rule : str, must be EN or L1, Optional
        Specifies which regularization must be used, either Elastic Net or L1.
    abort_early : bool, Optional
        Specifies if the attack should halt when stagnating  or not.

    References
    ----------
    .. [#Chen17] Chen, Pin-Yu, et al.
        "Ead: elastic-net attacks to deep neural networks via adversarial examples."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 32. No. 1. 2018.
    """
    __class_type = 'e-foolbox-ead'

    def __init__(self,
                 classifier: CClassifier,
                 y_target: Any = None,
                 lb=0.0,
                 ub=1.0,
                 epsilons=None,
                 binary_search_steps=9,
                 steps=50,
                 initial_stepsize=1e-2,
                 confidence=0.,
                 initial_const=1e-3,
                 regularization=1e-2,
                 decision_rule: str = EN,
                 abort_early=False,
                 ):
        if decision_rule != L1 and decision_rule != EN:
            raise ValueError(f"decision_rule param can be ony {EN} or {L1}, not {decision_rule}")
        super(CFoolboxEAD, self).__init__(classifier,
                                          y_target,
                                          lb=lb, ub=ub,
                                          fb_attack_class=_EADAttack,
                                          epsilons=epsilons,
                                          initial_const=initial_const,
                                          binary_search_steps=binary_search_steps,
                                          steps=steps,
                                          confidence=confidence,
                                          initial_stepsize=initial_stepsize,
                                          regularization=regularization,
                                          decision_rule=decision_rule,
                                          abort_early=abort_early)
        self.regularization = regularization
        self.confidence = confidence
        self.c = initial_const
        self._x0 = None
        self._y0 = None
        self.distance = 'l1'
        self._step_per_iter = None
        self.best_c_ = self.c

    def _run(self, x, y, x_init=None):
        self._x0 = as_tensor(x)
        self._y0 = as_tensor(y)
        out, _ = super(CFoolboxEAD, self)._run(x, y, x_init)
        self._consts = self.attack.consts
        self._f_seq: CArray = self.objective_function(self.x_seq)
        self.best_c_ = self._consts[self.attack._best_const]
        f_opt = self.objective_function(out)
        return out, f_opt

    @property
    def all_x_seq(self) -> list:
        divided_paths = self._slice_path()
        return divided_paths

    def _slice_path(self):
        all_paths = super(CFoolboxEAD, self).x_seq
        divided_paths = []
        for i, s in enumerate(self.attack._steps_per_iter):
            cumulative_sum = sum(self.attack._steps_per_iter[:i])
            divided_paths.append(all_paths[cumulative_sum: cumulative_sum + s, :])
        return divided_paths

    @property
    def x_seq(self):
        last_path = self._slice_path()[self.attack._best_const]
        return last_path


class _EADAttack(EADAttack):
    def run(
            self,
            model,
            inputs,
            criterion,
            *,
            early_stop=None,
            **kwargs: Any,
    ):
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        N = len(x)

        if isinstance(criterion_, Misclassification):
            targeted = False
            classes = criterion_.labels
            change_classes_logits = self.confidence
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes
            change_classes_logits = -self.confidence
        else:
            raise ValueError("unsupported criterion")

        def is_adversarial(perturbed: ep.Tensor, logits: ep.Tensor) -> ep.Tensor:
            if change_classes_logits != 0:
                logits += ep.onehot_like(logits, classes, value=change_classes_logits)
            return criterion_(perturbed, logits)

        if classes.shape != (N,):
            name = "target_classes" if targeted else "labels"
            raise ValueError(
                f"expected {name} to have shape ({N},), got {classes.shape}"
            )

        min_, max_ = model.bounds
        rows = range(N)

        def loss_fun(y_k: ep.Tensor, consts: ep.Tensor) -> Tuple[ep.Tensor, ep.Tensor]:
            assert y_k.shape == x.shape
            assert consts.shape == (N,)

            logits = model(y_k)

            if targeted:
                c_minimize = _best_other_classes(logits, classes)
                c_maximize = classes
            else:
                c_minimize = classes
                c_maximize = _best_other_classes(logits, classes)

            is_adv_loss = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert is_adv_loss.shape == (N,)

            is_adv_loss = is_adv_loss + self.confidence
            is_adv_loss = ep.maximum(0, is_adv_loss)
            is_adv_loss = is_adv_loss * consts

            squared_norms = ep.flatten(y_k - x).square().sum(axis=-1)
            loss = is_adv_loss.sum() + squared_norms.sum()
            return loss, logits

        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)

        consts = self.initial_const * ep.ones(x, (N,))
        lower_bounds = ep.zeros(x, (N,))
        upper_bounds = ep.inf * ep.ones(x, (N,))

        best_advs = ep.zeros_like(x)
        best_advs_norms = ep.ones(x, (N,)) * ep.inf

        self._consts = []
        self._steps_per_iter = []
        self._best_const = -1
        last_advs_norms = best_advs_norms

        # the binary search searches for the smallest consts that produce adversarials
        for binary_search_step in range(self.binary_search_steps):
            if (
                    binary_search_step == self.binary_search_steps - 1
                    and self.binary_search_steps >= 10
            ):
                # in the last iteration, repeat the search once
                consts = ep.minimum(upper_bounds, 1e10)

            # create a new optimizer find the delta that minimizes the loss
            x_k = x
            y_k = x
            iter_step = 0
            found_advs = ep.full(
                x, (N,), value=False
            ).bool()  # found adv with the current consts
            loss_at_previous_check = ep.inf

            for iteration in range(self.steps):
                # square-root learning rate decay
                stepsize = self.initial_stepsize * (1.0 - iteration / self.steps) ** 0.5

                loss, logits, gradient = loss_aux_and_grad(y_k, consts)

                x_k_old = x_k
                x_k = _project_shrinkage_thresholding(
                    y_k - stepsize * gradient, x, self.regularization, min_, max_
                )
                y_k = x_k + iteration / (iteration + 3.0) * (x_k - x_k_old)

                if self.abort_early and iteration % (math.ceil(self.steps / 10)) == 0:
                    # after each tenth of the iterations, check progress
                    if not loss.item() <= 0.9999 * loss_at_previous_check:
                        break  # stop optimization if there has been no progress
                    loss_at_previous_check = loss.item()
                iter_step += 1
                found_advs_iter = is_adversarial(x_k, model(x_k))

                best_advs, best_advs_norms = _apply_decision_rule(
                    self.decision_rule,
                    self.regularization,
                    best_advs,
                    best_advs_norms,
                    x_k,
                    x,
                    found_advs_iter,
                )

                if best_advs_norms < last_advs_norms:
                    self._best_const = binary_search_step
                    last_advs_norms = best_advs_norms

                found_advs = ep.logical_or(found_advs, found_advs_iter)
                self._consts.append(consts.numpy().tolist())

            self._steps_per_iter.append(iter_step)
            upper_bounds = ep.where(found_advs, consts, upper_bounds)
            lower_bounds = ep.where(found_advs, lower_bounds, consts)

            consts_exponential_search = consts * 10
            consts_binary_search = (lower_bounds + upper_bounds) / 2
            consts = ep.where(
                ep.isinf(upper_bounds), consts_exponential_search, consts_binary_search
            )

        return restore_type(best_advs)

    @property
    def consts(self):
        return CArray(self._consts).ravel()
