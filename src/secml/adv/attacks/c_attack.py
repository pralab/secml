"""
.. module:: CAttack
   :synopsis: Interface class for evasion and poisoning attacks.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta, abstractmethod, abstractproperty

from secml.core import CCreator
from secml.array import CArray
from secml.ml.classifiers import CClassifier


class CAttack(CCreator, metaclass=ABCMeta):
    """Generic interface class for adversarial attacks.

    Parameters
    ----------
    classifier : CClassifier
        Target classifier.

    """

    __super__ = 'CAttack'

    def __init__(self, classifier):
        # set the classifier to be attacked
        if not isinstance(classifier, CClassifier):
            raise ValueError("Classifier is not a CClassifier!")
        self._classifier = classifier

        # These are internal parameters populated by _run,
        # for the *last* attack point:
        self._x_opt = None  # the final/optimal attack point
        self._f_opt = None  # the objective value at the optimum
        self._x_seq = None  # the path of points through the optimization
        self._f_seq = None  # the objective values along the optimization path

    ###########################################################################
    #                           READ-ONLY ATTRIBUTES
    ###########################################################################

    @property
    def classifier(self):
        """Returns classifier"""
        return self._classifier

    @property
    def x_opt(self):
        """Returns the optimal point founded by the attack.

        Warnings
        --------
        Due to a known issue, if more then one sample is passed to ``.run()``,
        this property will only return the data relative to the last
        optimized one. This behavior will change in a future version.

        """
        return self._x_opt

    @property
    def f_opt(self):
        """
        Returns the value of the objective function evaluated on the optimal
        point founded by the attack.

        Warnings
        --------
        Due to a known issue, if more then one sample is passed to ``.run()``,
        this property will only return the data relative to the last
        optimized one. This behavior will change in a future version.

        """
        return self._f_opt

    @property
    def f_seq(self):
        """
        Returns a CArray containing the values of the objective function
        evaluations made by the attack.

        Warnings
        --------
        Due to a known issue, if more then one sample is passed to ``.run()``,
        this property will only return the data relative to the last
        optimized one. This behavior will change in a future version.

        """
        return self._f_seq

    @property
    def x_seq(self):
        """
        Returns a CArray (number of iteration * number of features) containing
        the values of the attack point path.

        Warnings
        --------
        Due to a known issue, if more then one sample is passed to ``.run()``,
        this property will only return the data relative to the last
        optimized one. This behavior will change in a future version.

        """
        return self._x_seq

    ###########################################################################
    #                       ABSTRACT PROPERTIES/METHODS
    ###########################################################################

    @property
    @abstractmethod
    def f_eval(self):
        """Returns the number of function evaluations made during the attack.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def grad_eval(self):
        """Returns the number of gradient evaluations made during the attack.

        """
        raise NotImplementedError

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
