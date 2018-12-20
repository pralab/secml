"""
Class CAttack

@author: Battista Biggio

Interface class for evasion and poisoning attacks.

"""
from abc import ABCMeta, abstractmethod, abstractproperty

from secml.core import CCreator
from secml.core.type_utils import is_int
from secml.array import CArray
from secml.ml.classifiers import CClassifier, CClassifierSVM
from secml.data import CDataset
from secml.data.splitter import CDataSplitter
# from secml.optimization.function import CFunction
# from secml.optimization.constraints import CConstraint
# from secml.adv.attacks.evasion.solvers import CSolver


class CAttack(CCreator):
    """
    Interface class for evasion and poisoning attacks.
    """
    __metaclass__ = ABCMeta
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
        """
        Initialization method.

        It requires classifier, surrogate_classifier, and surrogate_data.
        Note that surrogate_classifier is assumed to be trained (before
        passing it to this class) on surrogate_data.

        TODO: complete list of parameters

        Parameters
        ------
        discrete: True/False (default: false).
                  If True, input space is considered discrete (integer-valued),
                  otherwise continuous.
        attack_classes : 'all' or CArray, optional
            List of classes that can be manipulated by the attacker or
             'all' (default) if all classes can be manipulated.

        """

        # init attributes to None (replaced below using setters)
        self._classifier = None
        self._surrogate_classifier = None
        self._surrogate_data = None

        # labels and scores assigned by the surrogate clf to the surrogate data
        # these are "internal" attributes (no setter/getter), required to run
        # evasion of nonlinear clf from a benign point, and also to train a
        # surrogate differentiable classifier (if surrogate_clf is non-diff.)
        self._surrogate_labels = None
        self._surrogate_scores = None

        # set true/targeted classifier (and ndim)
        self.classifier = classifier

        # classes that can be manipulated by the attacker
        self._attack_classes = None

        # call setters
        self.attack_classes = attack_classes

        # init attributes to None (re-defined through setters below)
        self._solver = None

        # surrogate differentiable model used to optimize the attacks
        self._solver_clf = None

        # TODO: FULLY SUPPORT SPARSE DATA
        self._issparse = False  # We work with dense data by default
        self._dmax = None
        self._lb = None
        self._ub = None
        self._distance = None
        self._discrete = None
        self._y_target = None
        self._solver_type = None
        self._solver_params = None

        # now we populate solver parameters (via dedicated setters)
        self.dmax = dmax
        self.lb = lb
        self.ub = ub
        self.distance = distance
        self.discrete = discrete
        self.y_target = y_target
        self.solver_type = solver_type
        self.solver_params = solver_params

        # Setting mandatory attributes (besides classifier):
        # surrogate_classifier and surrogate_data.
        # The latter two allow us to define
        # solver_params['classifier'], which is passed to solver.
        # If any of them changes, solver is re-initialized.
        self.surrogate_classifier = surrogate_classifier

        # if surrogate_classifier is not differentiable,
        # surrogate_data can not be set to None.
        self.surrogate_data = surrogate_data

        CAttack.__clear(self)

    def __clear(self):
        """Reset the object."""
        # the attack point obtained after manipulation
        self._x_opt = None

        # value of objective function at x_opt
        self._f_opt = None

        # sequence of modifications to the attack point
        self._x_seq = None

        # ... and corresponding values of the objective function
        self._f_seq = None

        # number of function and gradient evaluations
        self._f_eval = 0
        self._grad_eval = 0

        # clear solver
        if self._solver is not None:
            self._solver.clear()

    def __is_clear(self):
        """Returns True if object is clear."""
        if self._x_opt is not None or self._f_opt is not None:
            return False
        if self._x_seq is not None or self._f_seq is not None:
            return False

        if self._solver is not None and not self._solver.is_clear():
            return False

        if self._f_eval + self._grad_eval != 0:
            return False

        return True

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

            for i in xrange(self.attack_classes.size):
                v[y == self.attack_classes[i]] = True  # y can be manipulated

            return v

        else:
            raise TypeError("y can be an integer or a CArray")

    ###########################################################################
    #                         ABSTRACT PUBLIC METHODS
    ###########################################################################

    @abstractmethod
    def run(self, x, y, ds_init=None):
        """
        Perform attack for the i-th param name attack power
        :param x:
        :param y:
        :param ds_init:
        :return:
        """
        raise ValueError("Not implemented!")

    @abstractmethod
    def _run(self, x, y, ds_init=None):
        """
        Move one single point for improve attacker objective function score
        :param x:
        :param y:
        :param ds_init:
        :return:
        """
        raise ValueError("Not implemented!")

    ###########################################################################
    #                           READ-ONLY ATTRIBUTES
    ###########################################################################

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
        return self._f_eval

    @property
    def grad_eval(self):
        return self._grad_eval

    ###########################################################################
    #                           READ-WRITE ATTRIBUTES
    ###########################################################################
    @property
    def classifier(self):
        """Returns classifier"""
        return self._classifier

    @classifier.setter
    def classifier(self, value):
        """Sets classifier"""
        if not isinstance(value, CClassifier):
            raise ValueError("Classifier is not a CClassifier!")

        self._classifier = value

    @property
    def surrogate_classifier(self):
        """Returns surrogate classifier"""
        return self._surrogate_classifier

    @surrogate_classifier.setter
    def surrogate_classifier(self, value):
        """Sets surrogate classifier"""
        if not isinstance(value, CClassifier):
            raise ValueError("Surrogate classifier is not a CClassifier!")

        self._surrogate_classifier = value

        # compute surrogate labels, if possible
        self._set_surrogate_labels_and_scores()

        # set classifier inside solver (and re-init solver)
        self._set_solver_classifier()

    @property
    def surrogate_data(self):
        """Returns surrogate data"""
        return self._surrogate_data

    @surrogate_data.setter
    def surrogate_data(self, value):
        """
        Sets surrogate data.
        If surrogate_classifier is differentiable,
        surrogate_data can be set to None,
        otherwise it has to be a CDataset.
        """
        if value is None and self._is_surrogate_clf_diff():
            self._surrogate_data = None
            return

        if not isinstance(value, CDataset):
            raise ValueError("Surrogate data is not a CDataset!")

        self._surrogate_data = value

        # compute surrogate labels, if possible
        self._set_surrogate_labels_and_scores()

        # if surrogate_data changes, and surrogate_classifier is non-diff.,
        # then a differentiable surrogate is learned and re-set inside solver.
        self._set_solver_classifier()

    ###########################################################################
    #            READ-WRITE ATTRIBUTES (from/to solver instance)
    ###########################################################################

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
    #                              PRIVATE METHODS
    ###########################################################################

    @abstractmethod
    def _objective_function(self, x):
        """
        Parameters
        ----------
        x: could be a matrix / dataset

        Returns
        -------
        f_obj: values of objective function at x
        """
        pass

    @abstractmethod
    def _objective_function_gradient(self, x):
        pass

    def _is_surrogate_clf_diff(self):
        """Returns True if the surrogate classifiers implements `gradient_f_x`.

        Returns
        -------
        bool
            True if the surrogate classifiers implements `gradient_f_x`,
            False otherwise.

        """
        try:  # Try to call gradient function with fake input
            self._surrogate_classifier.gradient_f_x(CArray([]), y=None)
        except NotImplementedError:
            return False  # Classifier does not implement the gradient
        except:  # Wildcard for any other error, gradient is implemented
            return True
        else:  # No error raised, gradient is implemented
            return True

    def _set_surrogate_labels_and_scores(self):
        """
        This function computes the labels assigned to the surrogate data
        by the surrogate classifier (only for nonlinear classifiers).
        """

        if self._surrogate_classifier is None or self._surrogate_data is None:
            self._surrogate_labels = None
            self._surrogate_scores = None
            return

        # otherwise...
        self.logger.info("Classification of surrogate data...")
        y, score = self.surrogate_classifier.classify(self.surrogate_data.X)
        self._surrogate_labels = y
        self._surrogate_scores = score

    def _set_solver_classifier(self):
        """This function returns the surrogate classifier,
        if differentiable; otherwise, it learns a smooth approximation for
        the nondiff. (surrogate) classifier (e.g., decision tree)
        using an SVM with the RBF kernel."""

        # check if surrogate learner is differentiable.
        if self._is_surrogate_clf_diff():
            self._solver_clf = self._surrogate_classifier
            return

        # if not, construct smooth approximation.
        # to this end, we need surrogate_data,
        # which may be not have been set yet
        if self._surrogate_data is None:
            return

        # train a differentiable surrogate classifier and pass it to solver
        self._solver_clf = self._train_differentiable_surrogate_clf()

        return

    # TODO: this should be customizable from outside of this class.
    def _train_differentiable_surrogate_clf(self):
        """
        Trains a differentiable surrogate classifier to be passed to the
        solver. By default, an RBF SVM is trained on surrogate data re-labeled
        by the surrogate (non-differentiable) classifier.
        """

        self.logger.info("Learning differentiable surrogate classifier...")

        # TODO: solve this more elegantly
        if self._surrogate_classifier.preprocess is None:
            preprocessor = None
        else:
            preprocessor = self._surrogate_classifier.preprocess.deepcopy()

        # creating instance of SVM learner
        clf = CClassifierSVM(kernel='rbf', preprocess=preprocessor)

        # clf.grad_sampling = 1 # speeding up gradient computation

        # construct dataset relabeled by surrogate learner
        relabeled_data = CDataset(
            self._surrogate_data.X,
            self._surrogate_labels)

        # configuring cross-validation to set C, gamma
        xval_parameters = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10]}
        xval_splitter = CDataSplitter.create('kfold', num_folds=5)
        xval_splitter.compute_indices(relabeled_data)

        # set best parameters
        best_params = clf.estimate_parameters(
            relabeled_data, xval_parameters, xval_splitter, 'accuracy')

        # train classifier with best params
        clf.set('C', best_params['C'])
        clf.set('gamma', best_params['gamma'])
        clf.train(relabeled_data)

        return clf

    # def _init_solver(self):
    #     """Create solver instance."""
    #
    #     if self._solver_clf is None or self.discrete is None:
    #         raise ValueError('Solver not set properly!')
    #
    #     # map attributes to fun, constr, box
    #     fun = CFunction(fun=self._objective_function,
    #                     gradient=self._objective_function_gradient,
    #                     n_dim=self.n_dim)
    #
    #     if self.solver_type is None:
    #         solver_type = 'descent-direction'
    #
    #     constr = None
    #     if self._distance is not None:
    #         constr = CConstraint.create(self._distance)
    #         constr.center = self._x0
    #         constr.radius = self.dmax
    #
    #     # only feature increments or decrements are allowed
    #     lb = self._x0 if self.lb == 'x0'else self.lb
    #     ub = self._x0 if self.ub == 'x0'else self.ub
    #     bounds = CConstraint.create('box', lb=lb, ub=ub)
    #
    #     self._solver = CSolver.create(
    #         solver_type,
    #         fun=fun, constr=constr,
    #         bounds=bounds,
    #         discrete=self._discrete,
    #         **self._solver_params)
    #
    #     # TODO: fix this verbose level propagation
    #     self._solver.verbose = self.verbose

    def _solution_from_solver(self):
        """
        Retrieve solution from solver and set internal class parameters.
        """
        self._f_eval += self._solver.f_eval
        self._grad_eval += self._solver.grad_eval

        # retrieve sequence of evasion points, and final point
        self._x_seq = self._solver.x_seq
        self._x_opt = self._solver.x_opt

        # retrieve f_obj values on x_seq and x_opt (from solver)
        self._f_seq = self._solver.f_seq
        self._f_opt = self._solver.f_opt
