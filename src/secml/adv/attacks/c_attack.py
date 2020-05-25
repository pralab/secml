"""
.. module:: CAttack
   :synopsis: Interface class for evasion and poisoning attacks.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta, abstractmethod

from secml.core import CCreator
from secml.core.type_utils import is_int
from secml.core.exceptions import NotFittedError
from secml.array import CArray
from secml.ml.classifiers import CClassifier
from secml.data import CDataset


class CAttack(CCreator, metaclass=ABCMeta):
    """Interface class for evasion and poisoning attacks."""

    __super__ = 'CAttack'

    @abstractmethod
    def run(self, x, y, ds_init=None):
        """Run attack on the dataset x,y (with multiple attack points).

        Parameters
        ----------
        x : CArray
            Initial samples.
        y : int or CArray
            The true label of x.
        ds_init : CDataset or None, optional.
            Dataset for warm start.

        Returns
        -------
        y_pred : predicted labels for all samples by the targeted classifier
        scores : scores for all samples by targeted classifier
        adv_ds : manipulated attack samples (for subsequents warm starts)
        f_opt : final value of the objective function

        """
        raise NotImplementedError

    @abstractmethod
    def _run(self, x, y):
        """Optimize the (single) attack point x,y.

        Parameters
        ----------
        x : CArray
            Sample.
        y : int or CArray
            The true label of x.

        """
        raise NotImplementedError

    @abstractmethod
    def objective_function(self, x):
        """Objective function.

        Parameters
        ----------
        x : CArray or CDataset

        Returns
        -------
        f_obj : float or CArray of floats

        """
        raise NotImplementedError

    @abstractmethod
    def objective_function_gradient(self, x):
        """Gradient of the objective function."""
        raise NotImplementedError
