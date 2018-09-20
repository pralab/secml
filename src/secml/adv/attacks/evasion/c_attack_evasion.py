"""
Class CEvasion

@author: Battista Biggio

This class performs evasion attacks
against a classifier, under different constraints.

"""

from secml.adv.attacks import CAttack
from secml.adv.attacks.evasion.solvers import CSolver
from secml.array import CArray
from secml.data import CDataset
from secml.core.constants import nan
from secml.optimization.function import CFunction
from secml.optimization.constraints import CConstraint
from secml.classifiers.loss import CLossSoftMax


class CAttackEvasion(CAttack):
    """
    Class that implements evasion attacks.
    """
    class_type = 'evasion'

    def __init__(self, classifier,
                 surrogate_classifier,
                 surrogate_data=None,
                 distance='l1',
                 dmax=0,
                 lb=0,
                 ub=1,
                 discrete=False,
                 y_target=None,
                 attack_classes=-1,
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
        attack_classes: list of classes that can be manipulated by the attacker
                 -1 means all classes can be manipulated.

        """
        self._x0 = None
        self._y0 = None

        # this is an alternative init point. See _get_point_with_min_f_obj()
        self._xk = None

        CAttack.__init__(self, classifier=classifier,
                         surrogate_classifier=surrogate_classifier,
                         surrogate_data=surrogate_data,
                         distance=distance,
                         dmax=dmax,
                         lb=lb,
                         ub=ub,
                         discrete=discrete,
                         y_target=y_target,
                         attack_classes=attack_classes,
                         solver_type=solver_type,
                         solver_params=solver_params)

    def __clear(self):
        self._x0 = None
        self._y0 = None
        self._xk = None

    ###########################################################################
    #                              PRIVATE METHODS
    ###########################################################################

    def _objective_function(self, x):
        """
        Parameters
        ----------
        x: is a single point!

        Returns
        -------
        f_obj: values of objective function at x
        """

        # Make classification in the sparse domain if possible
        x = x.tosparse() if self.issparse is True else x

        y_pred, scores = self._solver_clf.classify(x)
        return self._objective_function_scores(y_pred, scores)

    def _objective_function_scores(self, y_pred, scores):

        if self.y_target is None:
            # indiscriminate evasion
            # min f_obj(x) = f_k(x) - argmax_{c != k} f_c(x),
            # where k is the true class
            k = self._y0

            # the attacker aims to avoid the reject region,
            # to have the attack sample misclassified as a valid class.
            if hasattr(self._solver_clf, 'reject_option') and \
                    self._solver_clf.reject_option is True:
                scores[:, -1] = nan
        else:
            # targeted evasion
            # min -f_obj(x) =  -f_k(x) + argmax_{c != k} f_c(x),
            # where k is the target class
            k = self.y_target

        f_k = scores[:, k]
        scores[:, k] = nan
        f_obj = f_k - scores.nanmax(axis=1)

        return f_obj if self.y_target is None else -f_obj

    def _objective_function_gradient(self, x):

        # Make classification in the sparse domain if possible
        x = x.tosparse() if self.issparse is True else x

        y_pred, scores = self._solver_clf.classify(x)

        if self.y_target is None:
            # indiscriminate evasion
            # min f_obj(x) = f_k(x) - argmax_{c != k} f_c(x),
            # where k is the true class
            k = self._y0

            # the attacker aims to avoid the reject region,
            # to have the attack sample misclassified as a valid class.
            if hasattr(self._solver_clf, 'reject_option') and \
                    self._solver_clf.reject_option is True:
                scores[:, -1] = nan
        else:
            # targeted evasion
            # min -f_obj(x) =  -f_k(x) + argmax_{c != k} f_c(x),
            # where k is the target class
            k = self.y_target

        scores[:, k] = nan
        c = scores.nanargmax(axis=1)
        grad = self._solver_clf.gradient('x', x, y=k) - \
               self._solver_clf.gradient('x', x, y=c)

        return grad if self.y_target is None else -grad

    def _init_solver(self):
        """Create solver instance."""

        if self._solver_clf is None or self.distance is None \
                or self.discrete is None:
            raise ValueError('Solver not set properly!')

        # map attributes to fun, constr, box
        fun = CFunction(fun=self._objective_function,
                        gradient=self._objective_function_gradient,
                        n_dim=self.n_dim)

        solver_type = self._solver_type
        if solver_type is None:
            solver_type = 'descent-direction'

        constr = CConstraint.create(self._distance)
        constr.center = self._x0
        constr.radius = self.dmax

        # only feature increments or decrements are allowed
        lb = self._x0.todense() if self.lb == 'x0' else self.lb
        ub = self._x0.todense() if self.ub == 'x0' else self.ub

        bounds = CConstraint.create('box', lb=lb, ub=ub)

        self._solver = CSolver.create(
            solver_type,
            fun=fun, constr=constr,
            bounds=bounds,
            discrete=self._discrete,
            **self._solver_params)

        # TODO: fix this verbose level propagation
        self._solver.verbose = self.verbose

    # TODO: add probability as in c_attack_poisoning
    # (we could also move this direcly in c_attack)
    def _get_point_with_min_f_obj(self, y_pred, scores):
        """
        Retrieves the data point x with minimum value of objective function
        from surrogate data.

        Returns
        -------
        x: surrogate data point with minimum value of objective function
        """

        if self._surrogate_data is None:
            raise ValueError('Surrogate data has not been set!')

        f_obj = self._objective_function_scores(y_pred, scores)
        k = f_obj.argmin()
        return self._surrogate_data.X[k, :].ravel()

    # TODO: better to override setters for surrogate classifier and data
    # or use another mechanism
    def _set_solver_classifier(self):
        """Overriding to additionally compute xk."""
        super(CAttackEvasion, self)._set_solver_classifier()

        # TODO: better to isolate this function below from the rest
        # check if solver_clf and surrogate data are set
        if self._solver_clf is None or self._surrogate_data is None:
            return

        self.logger.info("Classification of surrogate data...")
        y_pred, scores = self._solver_clf.classify(self.surrogate_data.X)

        # for targeted evasion, this does not depend on the data label y0
        if self.y_target is not None:
            self._xk = self._get_point_with_min_f_obj(y_pred,
                                                      scores.deepcopy())
            return

        # for indiscriminate evasion, this depends on y0
        # so, we compute xk for all classes
        self._xk = CArray.zeros(shape=(self.surrogate_data.num_classes,
                                       self.surrogate_data.num_features),
                                sparse=self.surrogate_data.issparse,
                                dtype=self.surrogate_data.X.dtype)
        for i in xrange(self.surrogate_data.num_classes):
            self._y0 = i
            self._xk[i, :] = self._get_point_with_min_f_obj(y_pred,
                                                            scores.deepcopy())
        self._y0 = None
        return

    ###########################################################################
    #                              PUBLIC METHODS
    ###########################################################################

    # TODO: this should be private as _run
    def _run(self, x0, y0, x_init=None):
        """
        Perform evasion for a given dmax on a single pattern
        It solves:
            min_x g(x),
            s.t. c(x,x0) <= dmax

        Parameters:
        ------
        x0: initial malicious sample
        y0: the true label of x0
        x_init: init point (if None, set to x0)

        Returns:
        ------
        x_opt: evasion sample
        y_opt: classification label assigned to x_opt
        f_opt: value of objective function (from surrogate learner)

        Internally, this class stores the values of
        the objective function and sequence of attack points (if enabled).
        """
        self._f_eval = 0
        self._grad_eval = 0

        # if data can not be modified by the attacker, exit
        if not self._is_attack_class(y0):
            self._x_seq = x_init
            self._x_opt = x_init
            self._f_opt = nan
            self._f_seq = nan
            return self._x_opt, self._f_opt

        if x_init is None:
            x_init = x0

        if not isinstance(x_init, CArray):
            raise TypeError("Input vectors should be of class CArray")

        self._x0 = x0
        self._y0 = y0
        self._init_solver()

        # calling solver (on surrogate learner) and set solution variables
        self._solver.minimize(x_init)
        self._solution_from_solver()

        # if classifier is linear, or dmax is 0, return
        if self._classifier.is_linear() or self.dmax == 0:
            return self._x_opt, self._f_opt

        # value of objective function at x_opt
        f_obj = self._solver.f_opt

        # otherwise, try to improve evasion sample
        # we run an evasion attempt using (as the init sample)
        # the sample xk with the minimum objective function from surrogate data
        if self._xk is None:
            raise ValueError('xk not set!')

        # xk depends on whether evasion is targeted/indiscriminate
        xk = self._xk if self.y_target is not None else self._xk[self._y0, :]

        # if the projection of xk improves objective, try to restart
        xk_proj = self._solver._constr.projection(xk)

        # TODO: this has to be fixed
        # discretize x_min_proj on grid
        # xk_proj = (xk_proj / self._solver.eta).round() * self._solver.eta

        if self._objective_function(xk_proj) < f_obj:

            self.logger.debug("Trying to improve current solution.")

            self._solver.minimize(xk)
            f_obj_min = self._solver.f_opt

            # if this solution is better than the previous one,
            # we use the current solution found by the solver
            if f_obj_min < f_obj:
                self.logger.info("Better solution from restarting point!")
                self._solution_from_solver()

        return self._x_opt, self._f_opt

    # TODO: this should run either on a single x,y point, or on multiple points
    # and also return f_opt or its average (define a common interface also for
    # poisoning attacks)
    def run(self, X, Y, ds_init=None):
        """
        Runs evasion on a dataset.

        Parameters
        ----------
        x: data points (more than one)
        y: true labels
        ds_init: for warm starts

        Returns
        -------
        y_pred: predicted labels for all ds samples by targeted classifier
        scores: scores for all ds samples by targeted classifier
        adv_ds: manipulated ds sample dataset
        """
        x = CArray(X).atleast_2d()
        y = CArray(Y).atleast_2d()
        x_init = None if ds_init is None else CArray(ds_init.X).atleast_2d()

        # only consider samples that can be manipulated
        v = self.is_attack_class(y)
        idx = CArray(CArray(v.find(v)).ravel())  # TODO: fix CArray
        # print v, idx

        # number of modifiable samples
        n_mod_samples = idx.size

        adv_ds = CDataset(x.deepcopy(), y.deepcopy())

        # TODO: avoid storing sample as we have them already in adv_ds
        first_eva = []  # None or x_opt, if evaded

        # If dataset is sparse, set the proper attribute
        if X.issparse is True:
            self._issparse = True

        for i in xrange(n_mod_samples):
            k = idx[i]  # idx of sample that can be modified

            xi = x[k, :] if x_init is None else x_init[k, :]
            x_opt, f_opt = self._run(x[k, :], y[k], x_init=xi)

            self.logger.info(
                "Point: {:}/{:}, dmax:{:}, f(x):{:}, eval:{:}/{:}".format(
                    k, x.shape[0], self._dmax, f_opt, self._f_eval,
                    self._grad_eval))
            adv_ds.X[k, :] = x_opt

        y_pred, scores = self.classifier.classify(adv_ds.X)

        y_pred = CArray(y_pred)

        for i in xrange(adv_ds.num_samples):
            if (self.y_target is not None and y_pred[i] == self.y_target) or \
                    (self.y_target is None and y_pred[i] != y[i]):
                first_eva.append(adv_ds.X[i, :])
            else:
                first_eva.append(None)

        # Compute the attacker objective function
        f_obj = CLossSoftMax().loss(y.ravel(), scores, c=self.y_target).mean()

        return y_pred, scores, adv_ds, f_obj, first_eva
