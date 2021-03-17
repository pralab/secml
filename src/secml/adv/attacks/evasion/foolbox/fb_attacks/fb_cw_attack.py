"""
.. module:: CFoolboxL2CarliniWagner
    :synopsis: Performs Foolbox CW L2 attack.

.. moduleauthor:: Luca Demetrio <luca.demetrio@dibris.unige.it>
.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""

from functools import partial

import eagerpy as ep
import foolbox as fb
import numpy as np
from foolbox import Misclassification, TargetedMisclassification
from foolbox.attacks.base import raise_if_kwargs, get_criterion
from foolbox.attacks.carlini_wagner import _to_attack_space, _to_model_space, best_other_classes, AdamOptimizer
from foolbox.devutils import flatten, atleast_kd

from secml.adv.attacks.evasion.foolbox.c_attack_evasion_foolbox import CAttackEvasionFoolbox
from secml.adv.attacks.evasion.foolbox.losses.cw_loss import CWLoss
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor
from secml.array import CArray


class CFoolboxL2CarliniWagner(CWLoss, CAttackEvasionFoolbox):
    """
    Carlini & Wagner L2 Attack [#Carl16]_.

    Credits: https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/carlini_wagner.py

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
    binary_search_steps : int, optional
        The number of steps to perform in the binary search over the constant c.
    steps : int, optional
        Number of update steps to perform within each binary search step.
    stepsize : float, optional
        Stepsize to update the examples.
    confidence : float, optional
        Confidence required to mark an example as adversarial.
        Controls the gap between decision boundary and adversarial example.
    initial_const : float, optional
        Initial value of the constant c when the binary search starts.
    abort_early : bool, optional
        Stop inner search when an adversarial example has been found.
        It does not affect the binary search.

    References
    ----------
    .. [#Carl16] Nicholas Carlini, David Wagner, "Towards evaluating the robustness of
        neural networks. In 2017 ieee symposium on security and privacy"
        https://arxiv.org/abs/1608.04644
    """
    __class_type = 'e-foolbox-cw'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0,
                 binary_search_steps=9, steps=10000, stepsize=1e-2,
                 confidence=0, initial_const=1e-3, abort_early=True):
        super(CFoolboxL2CarliniWagner, self).__init__(classifier, y_target,
                                                      lb=lb, ub=ub,
                                                      fb_attack_class=_L2CarliniWagnerAttack,
                                                      epsilons=None,
                                                      binary_search_steps=binary_search_steps,
                                                      steps=steps, stepsize=stepsize,
                                                      confidence=confidence,
                                                      initial_const=initial_const,
                                                      abort_early=abort_early)
        self.confidence = confidence
        self.c = initial_const
        self._x0 = None
        self._y0 = None
        self.distance = 'l2'
        self._step_per_iter = None
        self.best_c_ = self.c

    def _run(self, x, y, x_init=None):
        self._x0 = as_tensor(x)
        self._y0 = as_tensor(y)
        out, _ = super(CFoolboxL2CarliniWagner, self)._run(x, y, x_init)
        self._consts = self.attack.consts
        self._f_seq = self.objective_function(self.x_seq)
        self.best_c_ = self._consts[self.attack._best_const]
        f_opt = self.objective_function(out)
        return out, f_opt

    @property
    def all_x_seq(self) -> list:
        divided_paths = self._slice_path()
        return divided_paths

    def _slice_path(self):
        all_paths = super(CFoolboxL2CarliniWagner, self).x_seq
        divided_paths = []
        for i, s in enumerate(self.attack._steps_per_iter):
            cumulative_sum = sum(self.attack._steps_per_iter[:i])
            divided_paths.append(all_paths[cumulative_sum: cumulative_sum + s, :])
        return divided_paths

    @property
    def x_seq(self):
        last_path = self._slice_path()[self.attack._best_const]
        return last_path


class _L2CarliniWagnerAttack(fb.attacks.L2CarliniWagnerAttack):
    def run(self, model, inputs, criterion, *, early_stop, **kwargs):
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

        bounds = model.bounds
        to_attack_space = partial(_to_attack_space, bounds=bounds)
        to_model_space = partial(_to_model_space, bounds=bounds)

        x_attack = to_attack_space(x)
        reconstsructed_x = to_model_space(x_attack)

        rows = range(N)

        def loss_fun(delta, consts):
            assert delta.shape == x_attack.shape
            assert consts.shape == (N,)

            x = to_model_space(x_attack + delta)
            logits = model(x)

            if targeted:
                c_minimize = best_other_classes(logits, classes)
                c_maximize = classes  # target_classes
            else:
                c_minimize = classes  # labels
                c_maximize = best_other_classes(logits, classes)

            is_adv_loss = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert is_adv_loss.shape == (N,)

            is_adv_loss = is_adv_loss + self.confidence
            is_adv_loss = ep.maximum(0, is_adv_loss)
            is_adv_loss = is_adv_loss * consts

            squared_norms = flatten(x - reconstsructed_x).square().sum(axis=-1)
            loss = is_adv_loss.sum() + squared_norms.sum()
            return loss, (x, logits)

        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)

        consts = self.initial_const * np.ones((N,))
        lower_bounds = np.zeros((N,))
        upper_bounds = np.inf * np.ones((N,))

        best_advs = ep.zeros_like(x)
        best_advs_norms = ep.full(x, (N,), ep.inf)

        self._consts = []
        self._steps_per_iter = []
        self._best_const = -1
        # the binary search searches for the smallest consts that produce adversarials
        for binary_search_step in range(self.binary_search_steps):
            if (
                    binary_search_step == self.binary_search_steps - 1
                    and self.binary_search_steps >= 10
            ):
                # in the last binary search step, repeat the search once
                consts = np.minimum(upper_bounds, 1e10)

            iter_step = 0

            # create a new optimizer find the delta that minimizes the loss
            delta = ep.zeros_like(x_attack)
            optimizer = AdamOptimizer(delta)

            # tracks whether adv with the current consts was found
            found_advs = np.full((N,), fill_value=False)
            loss_at_previous_check = np.inf

            consts_ = ep.from_numpy(x, consts.astype(np.float32))

            for step in range(self.steps):
                loss, (perturbed, logits), gradient = loss_aux_and_grad(delta, consts_)
                delta += optimizer(gradient, self.stepsize)

                if self.abort_early and step % (np.ceil(self.steps / 10)) == 0:
                    # after each tenth of the overall steps, check progress
                    if not (loss <= 0.9999 * loss_at_previous_check):
                        break  # stop Adam if there has been no progress
                    loss_at_previous_check = loss

                iter_step += 1

                found_advs_iter = is_adversarial(perturbed, logits)
                found_advs = np.logical_or(found_advs, found_advs_iter.numpy())

                norms = flatten(perturbed - x).norms.l2(axis=-1)
                closer = norms < best_advs_norms
                new_best = ep.logical_and(closer, found_advs_iter)
                if closer and found_advs_iter:
                    self._best_const = binary_search_step

                new_best_ = atleast_kd(new_best, best_advs.ndim)
                best_advs = ep.where(new_best_, perturbed, best_advs)
                best_advs_norms = ep.where(new_best, norms, best_advs_norms)
                self._consts.append(consts_.numpy().tolist())

            self._steps_per_iter.append(iter_step)

            upper_bounds = np.where(found_advs, consts, upper_bounds)
            lower_bounds = np.where(found_advs, lower_bounds, consts)

            consts_exponential_search = consts * 10
            consts_binary_search = (lower_bounds + upper_bounds) / 2
            consts = np.where(
                np.isinf(upper_bounds), consts_exponential_search, consts_binary_search
            )

        return restore_type(best_advs)

    @property
    def consts(self):
        return CArray(self._consts).ravel()
