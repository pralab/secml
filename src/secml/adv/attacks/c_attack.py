"""
.. module:: CAttack
   :synopsis: Interface class for evasion and poisoning attacks.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta, abstractmethod
import six
from six.moves import range

from secml.core import CCreator
from secml.core.type_utils import is_int
from secml.core.exceptions import NotFittedError
from secml.array import CArray
from secml.ml.classifiers import CClassifier
from secml.data import CDataset


@six.add_metaclass(ABCMeta)
class CAttack(CCreator):
    """Interface class for evasion and poisoning attacks.

    Parameters
    ----------
    classifier : CClassifier
        Target classifier.
    surrogate_classifier : CClassifier
        Surrogate classifier, assumed to be already trained.
    surrogate_data : CDataset or None, optional
        Dataset on which the the surrogate classifier has been trained on.
        Is only required if the classifier is nonlinear.
    distance : {'l1' or 'l2'}, optional
        Norm to use for computing the distance of the adversarial example
        from the original sample. Default 'l2'.
    dmax : scalar, optional
        Maximum value of the perturbation. Default 1.
    lb, ub : int or CArray, optional
        Lower/Upper bounds. If int, the same bound will be applied to all
        the features. If CArray, a different bound can be specified for each
        feature. Default `lb = 0`, `ub = 1`.
    discrete: True/False (default: false).
        If True, input space is considered discrete (integer-valued),
        otherwise continuous.
    y_target : int or None, optional
        If None an error-generic attack will be performed, else a
        error-specific attack to have the samples misclassified as
        belonging to the `y_target` class.
    attack_classes : 'all' or CArray, optional
        Array with the classes that can be manipulated by the attacker or
         'all' (default) if all classes can be manipulated.
    solver_type : str or None, optional
        Identifier of the solver to be used.
    solver_params : dict or None, optional
        Parameters for the solver. Default None, meaning that default
        parameters will be used.

    """
    __super__ = 'CAttack'

    def __init__(self, classifier,
                 surrogate_classifier,
                 surrogate_data=None,
                 distance=None,
                 dmax=None,
                 lb=None,
                 ub=None,
                 discrete=False,
                 y_target=None,
                 attack_classes='all',
                 solver_type=None,
                 solver_params=None):

        # INTERNAL
        # init attributes to None (re-defined through setters below)
        self._solver = None
        # surrogate differentiable model used to optimize the attacks
        self._solver_clf = None

        # READ ONLY ATTRIBUTES

        # the attack point obtained after manipulation
        self._x_opt = None
        # value of objective function at x_opt
        self._f_opt = None
        # sequence of modifications to the attack point
        self._x_seq = None
        # ... and corresponding values of the objective function
        self._f_seq = None
        # TODO: FULLY SUPPORT SPARSE DATA
        self._issparse = False  # We work with dense data by default

        # set true/targeted classifier (and ndim)
        if not isinstance(classifier, CClassifier):
            raise ValueError("Classifier is not a CClassifier!")
        self._classifier = classifier

        # READ/WRITE

        # classes that can be manipulated by the attacker
        self.attack_classes = attack_classes

        # now we populate solver parameters (via dedicated setters)
        self.dmax = dmax
        self.lb = lb
        self.ub = ub
        self.distance = distance
        self.discrete = discrete
        self.y_target = y_target
        self.solver_type = solver_type
        self.solver_params = solver_params

        # Surrogate classifier. Should be differentiable
        self.surrogate_classifier = surrogate_classifier

        # Surrogate data. Required in case of a nonlinear surrogate classifier
        self.surrogate_data = surrogate_data

    ###########################################################################
    #                           READ-ONLY ATTRIBUTES
    ###########################################################################

    @property
    def classifier(self):
        """Returns classifier"""
        return self._classifier

    @property
    def n_dim(self):  # dimensionality of x0
        if self._solver_clf is None:
            raise ValueError('Classifier is not set.')
        return self._solver_clf.n_features

    @property
    def issparse(self):
        return self._issparse

    @property
    def x_opt(self):
        return self._x_opt

    @property
    def f_opt(self):
        return self._f_opt

    @property
    def f_seq(self):
        return self._f_seq

    @property
    def x_seq(self):
        return self._x_seq

    @property
    def f_eval(self):
        return self._solver.f_eval

    @property
    def grad_eval(self):
        return self._solver.grad_eval

    ###########################################################################
    #                           READ-WRITE ATTRIBUTES
    ###########################################################################

    # TODO: REMOVE SETTERS FOR surrogate_classifier and surrogate_data as
    #  they are required init attributes (resetting them must require reinit)

    @property
    def surrogate_classifier(self):
        """Returns surrogate classifier"""
        return self._surrogate_classifier

    @surrogate_classifier.setter
    def surrogate_classifier(self, clf):
        """Sets surrogate classifier"""
        if not isinstance(clf, CClassifier):
            raise ValueError("Surrogate classifier is not a CClassifier!")

        # TODO: WE DO NOT CURRENTLY HAVE A RELIABLE WAY TO CHECK IF THE
        #  CLASSIFIER IS DIFFERENTIABLE. `_run` WILL CRASH ANYWAY LATER
        #  IF `_gradient_f` IS NOT DEFINED

        # TODO: MAYBE WE CAN REMOVE THIS? AN ERROR WILL BE RAISED LATER ANYWAY
        if not clf.is_fitted():
            raise NotFittedError(
                "the surrogate classifier must be already trained")

        self._surrogate_classifier = clf

        # set classifier inside solver (and re-init solver)
        self._set_solver_classifier()

    @property
    def surrogate_data(self):
        """Returns surrogate data"""
        return self._surrogate_data

    @surrogate_data.setter
    def surrogate_data(self, value):
        """Sets surrogate data."""
        if not isinstance(value, CDataset):
            raise ValueError("Surrogate data is not a CDataset!")
        self._surrogate_data = value

        # Surrogate data changed, reset predictions of solver classifier
        self._clear_solver_surrogate_predictions()

    @property
    def attack_classes(self):
        return self._attack_classes

    @attack_classes.setter
    def attack_classes(self, values):
        if not (values == 'all' or isinstance(values, CArray)):
            raise ValueError("`attack_classes` can be 'all' or a CArray")
        self._attack_classes = values

    def is_attack_class(self, y):
        """Returns True/False if the input class can be attacked.

        Parameters
        ----------
        y : int or CArray
            CArray or single label of the class to to be checked.

        Returns
        -------
        bool or CArray
            True if class y can be manipulated by the attacker,
             False otherwise. If CArray, a True/False value for each
             input label will be returned.

        """
        if is_int(y):
            if self._attack_classes == 'all':
                return True  # all classes can be manipulated
            elif CArray(y == self._attack_classes).any():
                return True  # y can be manipulated
            else:
                return False

        elif isinstance(y, CArray):

            v = CArray.zeros(shape=y.shape, dtype=bool)

            if self.attack_classes == 'all':
                v[:] = True  # all classes can be manipulated
                return v

            for i in range(self.attack_classes.size):
                v[y == self.attack_classes[i]] = True  # y can be manipulated

            return v

        else:
            raise TypeError("y can be an integer or a CArray")

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
    def discrete(self):
        """Returns True if feature space is discrete, False if continuous."""
        return self._discrete

    @discrete.setter
    def discrete(self, value):
        """Set to True if feature space is discrete, False if continuous."""
        if value is None:
            self._discrete = None
            return

        self._discrete = bool(value)

    @property
    def y_target(self):
        return self._y_target

    @y_target.setter
    def y_target(self, value):
        self._y_target = value

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

    @abstractmethod
    def run(self, x, y, ds_init=None):
        """Perform attack for the i-th param name attack power.

        Parameters
        ----------
        x : CArray
            Initial sample.
        y : int or CArray
            The true label of x.
        ds_init : CDataset or None, optional.
            Dataset for warm start.

        """
        raise NotImplementedError

    @abstractmethod
    def _run(self, x, y):
        """Move one single point for improve attacker objective function score.

        Parameters
        ----------
        x : CArray
            Sample.
        y : int or CArray
            The true label of x.

        """
        raise NotImplementedError

    @abstractmethod
    def _objective_function(self, x):
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
    def _objective_function_gradient(self, x):
        """Gradient of the objective function."""
        raise NotImplementedError

    def _set_solver_classifier(self):
        """Sets the classifier of the solver."""
        self._solver_clf = self._surrogate_classifier

        # Solver classifier changed, reset predictions on surrogate dataset
        self._clear_solver_surrogate_predictions()

    def _set_solver_surrogate_predictions(self):
        """Compute predictions on surrogate data using solver classifier."""
        if self.surrogate_data is None:
            raise ValueError("surrogate data is not defined")

        # Reset the current predictions
        self._clear_solver_surrogate_predictions()

        # Compute the new predictions
        self.logger.info("Classification of surrogate data...")
        y, score = self._solver_clf.predict(
            self.surrogate_data.X, return_decision_function=True)
        self._surrogate_labels = y
        self._surrogate_scores = score

    def _clear_solver_surrogate_predictions(self):
        """Reset the predictions on surrogate data using solver classifier."""
        self._surrogate_labels = None
        self._surrogate_scores = None

    def _solution_from_solver(self):
        """Retrieve solution from solver and set internal class parameters."""
        # retrieve sequence of evasion points, and final point
        self._x_seq = self._solver.x_seq
        self._x_opt = self._solver.x_opt

        # retrieve f_obj values on x_seq and x_opt (from solver)
        self._f_seq = self._solver.f_seq
        self._f_opt = self._solver.f_opt
