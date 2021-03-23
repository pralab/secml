"""
.. module:: CAttackEvasionFoolbox
    :synopsis: Performs one of the Foolbox Evasion attacks
                against a classifier.

.. moduleauthor:: Luca Demetrio <luca.demetrio@dibris.unige.it>
.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""
import eagerpy as ep
import foolbox as fb
import torch
from eagerpy import PyTorchTensor
from numpy import NaN

from secml.adv.attacks.evasion import CAttackEvasion
from secml.adv.attacks.evasion.foolbox.secml_autograd import \
    SecmlLayer, as_tensor, as_carray
from secml.array import CArray
from secml.core.constants import inf
from secml.settings import SECML_PYTORCH_USE_CUDA

use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA


class CAttackEvasionFoolbox(CAttackEvasion):
    """
    Wrapper for the attack classes in Foolbox library.

    Credits: https://foolbox.readthedocs.io/en/stable/.
    Requires foolbox >= 3.3.0.

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
    fb_attack_class : fb.attacks.Attack
        Attack class to wrap from Foolbox.
    **attack_params : any
        Init parameters for creating the attack, as kwargs.
    """

    __class_type = 'e-foolbox'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0,
                 epsilons=None, fb_attack_class=None, **attack_params):

        super(CAttackEvasionFoolbox, self).__init__(
            classifier=classifier,
            y_target=y_target)

        self.attack_params = attack_params
        self.attack_class = fb_attack_class

        self.lb = lb
        self.ub = ub

        # wraps secml classifier in a pytorch layer
        self._pytorch_model_wrapper = SecmlLayer(classifier)
        # wraps the pytorch model in the foolbox pytorch wrapper
        self.f_model = _FoolboxModel(self._pytorch_model_wrapper,
                                     bounds=(lb, ub))

        self._last_f_eval = None
        self._last_grad_eval = None

        self._n_classes = self.classifier.n_classes
        self._n_feats = self.classifier.n_features

        self.epsilon = epsilons
        self.dmax = epsilons if epsilons is not None else inf

        self.attack = self.attack_class(**self.attack_params)

    def _run(self, x, y, x_init=None):
        self.f_model.reset()
        if self.y_target is None:
            criterion = fb.criteria.Misclassification(
                as_tensor(y.ravel().astype('int64')))
        else:
            criterion = fb.criteria.TargetedMisclassification(
                torch.tensor([self.y_target]))

        x_t = as_tensor(x, requires_grad=False)
        advx, clipped, is_adv = self.attack(
            self.f_model, x_t, criterion, epsilons=self.epsilon)

        if isinstance(clipped, list):
            if len(clipped) == 1:
                clipped = x[0]
            else:
                raise ValueError(
                    "This attack is returning a list. Please,"
                    "use a single value of epsilon.")

        # f_opt is computed only in class-specific wrappers
        f_opt = NaN

        self._last_f_eval = self.f_model.f_eval
        self._last_grad_eval = self.f_model.grad_eval
        path = self.f_model.x_path
        self._x_seq = CArray(path.numpy())

        # reset again to clean cached data
        self.f_model.reset()
        return as_carray(clipped), f_opt

    def objective_function(self, x):
        return as_carray(self._adv_objective_function(as_tensor(x)))

    def objective_function_gradient(self, x):
        x_t = as_tensor(x).detach()
        x_t.requires_grad_()
        loss = self._adv_objective_function(x_t)
        loss.sum().backward()
        gradient = x_t.grad
        return as_carray(gradient)

    def _adv_objective_function(self, x):
        raise NotImplementedError(
            "Objective Function and Objective Function Gradient "
            "are not supported with this constructor. Please, "
            "use one of our wrapper-supported attacks.")

    @property
    def x_seq(self):
        return self._x_seq

    @property
    def f_eval(self):
        if self._last_f_eval is not None:
            return self._last_f_eval
        else:
            raise RuntimeError("Attack not run yet!")

    @property
    def grad_eval(self):
        if self._last_grad_eval is not None:
            return self._last_grad_eval
        else:
            raise RuntimeError("Attack not run yet!")


class _FoolboxModel(fb.models.PyTorchModel):
    """Wraps a model and tracks function calls."""

    def __init__(self, model, bounds, store_path=True):
        self._original_model = model
        self._f_eval = 0
        self._grad_eval = 0
        self._store_path = store_path
        self._x_path = []
        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                "expected model to be a torch.nn.Module instance")

        device = 'cuda' if use_cuda else 'cpu'
        super().__init__(
            model, bounds=bounds, preprocessing=None, device=device,
        )

        self.data_format = "channels_first"

    @property
    def bounds(self):
        return self._bounds

    @property
    def x_path(self):
        path = ep.concatenate(self._x_path, axis=0)
        return path[:-1, ...]  # removes last point

    @property
    def f_eval(self):
        return self._original_model.func_counter.item()

    @property
    def grad_eval(self):
        return self._original_model.grad_counter.item()

    def __call__(self, x, *args, **kwargs):
        x_t = x.raw.type(torch.float)
        scores = self._model(x_t)
        if self._store_path is True:
            self._x_path.append(x)
        return PyTorchTensor(scores)

    def reset(self):
        """Resets the query counter."""
        self._original_model.func_counter.zero_()
        self._original_model.grad_counter.zero_()
        if self._store_path is True:
            self._x_path = list()
