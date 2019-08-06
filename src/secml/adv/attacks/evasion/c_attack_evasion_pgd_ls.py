"""
.. module:: CAttackEvasionPGDLS
   :synopsis: Evasion attack using Projected Gradient Descent with Bisect Line Search.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from six.moves import range

from secml.adv.attacks import CAttack
from secml.adv.attacks.evasion import CAttackEvasion
from secml.optim.optimizers import COptimizer
from secml.array import CArray
from secml.core.constants import nan
from secml.optim.function import CFunction
from secml.optim.constraints import CConstraint
from secml.ml.classifiers.reject import CClassifierReject


class CAttackEvasionPGDLS(CAttackEvasion):
    """Evasion attacks using Projected Gradient Descent with Line Search.

    This class implements the maximum-confidence evasion attacks proposed in:
     - https://arxiv.org/abs/1708.06939, ICCV W. ViPAR, 2017.

    This is the multi-class extension of our original work in:
     - https://arxiv.org/abs/1708.06131, ECML 2013,

    implemented using a custom projected gradient solver that uses line search
    in each iteration to save gradient computations and speed up the attack.

    It can also be used on sparse, high-dimensional feature spaces, using an
    L1 constraint on the manipulation of samples to preserve sparsity,
    as we did for crafting adversarial Android malware in:
     - https://arxiv.org/abs/1704.08996, IEEE TDSC 2017.

    For more on evasion attacks, see also:
     - https://arxiv.org/abs/1809.02861, USENIX Sec. 2019
     - https://arxiv.org/abs/1712.03141, Patt. Rec. 2018

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
    solver_params : dict or None, optional
        Parameters for the solver. Default None, meaning that default
        parameters will be used.

    Attributes
    ----------
    class_type : 'e-pgd-ls'

    """
    __class_type = 'e-pgd-ls'

    def __init__(self, classifier,
                 surrogate_classifier,
                 surrogate_data=None,
                 distance='l1',
                 dmax=0,
                 lb=0,
                 ub=1,
                 discrete=False,
                 y_target=None,
                 attack_classes='all',
                 solver_params=None):

        # INTERNALS
        self._x0 = None
        self._y0 = None

        # this is an alternative init point. This could be a single point
        # (targeted evasion) or an array of multiple points, one for each
        # class (indiscriminate evasion). See _get_point_with_min_f_obj()
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
                         solver_type='pgd-ls',
                         solver_params=solver_params)

    ###########################################################################
    #                           READ-WRITE ATTRIBUTES
    ###########################################################################

    # OVERRIDE y_target to reset the alternative init point xk
    @property
    def y_target(self):
        return self._y_target

    @y_target.setter
    def y_target(self, value):
        self._y_target = value
        self._xk = None

    ###########################################################################
    #                              PRIVATE METHODS
    ###########################################################################

    def _find_k_c(self, y_pred, scores):
        """Find the class of which we aim to maximize and the one of which we
         aim to minimize the score.

        This function works on the prediction and score of either, a single
        or multiple samples.

        """
        scores = scores.deepcopy()

        n_samples = y_pred.size

        k = CArray.zeros(shape=(n_samples,), dtype=int)

        if self.y_target is None:  # indiscriminate attack

            # if the sample is not rejected k is the true class
            k[:] = self._y0

            # c is neither k nor the reject class
            smpls_idx = CArray.arange(n_samples).tolist()

            # set to nan the score of the true classes to exclude it by
            # the successive choice of the competing classes
            scores[[smpls_idx, k.tolist()]] = nan

            if issubclass(self._solver_clf.__class__, CClassifierReject):
                # set to nan the score of the reject classes to exclude it by
                # the successive choice of the competing classes
                scores[:, -1] = nan

            # for the rejected samples k is the reject class
            k[y_pred == -1] = -1

        else:  # targeted attack

            # c is not the target class
            scores[:, self.y_target] = nan

            # k is the target class
            k[:] = self.y_target

        c = scores.nanargmax(axis=1).ravel()

        if issubclass(self._solver_clf.__class__, CClassifierReject):
            c[c == self.surrogate_data.num_classes] = -1

        return k, c

    def _objective_function(self, x):
        """Compute the objective function of the evasion attack.

        The objective function is:

        - for error-generic attack:
            min f_obj(x) = f_{k|o (if the sample is rejected) }(x)
            argmax_{(c != k) and (c != o)} f_c(x),
            where k is the true class, o is the reject class and c is the
            competing class, which is the class with the maximum score, and
            can be neither k nor c

        -for error-specific attack:
            min -f_obj(x) =  -f_k(x) + argmax_{c != k} f_c(x),
            where k is the target class and c is the competing class,
            which is the class with the maximum score except for the
            target class

        Parameters
        ----------
        x : CArray
            Array containing the data points (one or more than one).

        Returns
        -------
        f_obj : CArray
            Values of objective function at x.

        """
        # Make classification in the sparse domain if possible
        x = x.tosparse() if self.issparse is True else x

        y_pred, scores = self._solver_clf.predict(
            x, return_decision_function=True)

        f_obj = self._objective_function_pred_scores(y_pred, scores)

        return f_obj

    def _objective_function_pred_scores(self, y_pred, scores):
        """
        Given the predicted labels and the scores, compute the objective
        function. (This function allows to use already computed prediction
        labels and scores)
        """
        n_samples = y_pred.size

        k, c = self._find_k_c(y_pred, scores)

        smpls_idx = CArray.arange(n_samples).tolist()
        f_k = scores[[smpls_idx, k.tolist()]]
        f_obj = f_k - scores[[smpls_idx, c.tolist()]]

        return f_obj if self.y_target is None else -f_obj

    def _objective_function_gradient(self, x):
        """Compute the gradient of the evasion objective function.

        Parameters
        ----------
        x : CArray
            A single point.

        """
        # Make classification in the sparse domain if possible
        x = x.tosparse() if self.issparse is True else x

        y_pred, scores = self._solver_clf.predict(
            x, return_decision_function=True)

        k, c = self._find_k_c(y_pred, scores)

        grad = self._solver_clf.grad_f_x(x, y=k.item()) - \
               self._solver_clf.grad_f_x(x, y=c.item())

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

        constr = CConstraint.create(self._distance)
        constr.center = self._x0
        constr.radius = self.dmax

        # only feature increments or decrements are allowed
        lb = self._x0.todense() if self.lb == 'x0' else self.lb
        ub = self._x0.todense() if self.ub == 'x0' else self.ub

        bounds = CConstraint.create('box', lb=lb, ub=ub)

        self._solver = COptimizer.create(
            self._solver_type,
            fun=fun, constr=constr,
            bounds=bounds,
            discrete=self._discrete,
            **self._solver_params)

        # TODO: fix this verbose level propagation
        self._solver.verbose = self.verbose

    # TODO: add probability as in c_attack_poisoning
    # (we could also move this directly in c_attack)
    def _get_point_with_min_f_obj(self, y_pred, scores):
        """Returns the surrogate sample having the minimum value of objective function.

        Parameters
        ----------
        y_pred : CArray
            Predictions on surrogate data of the solver classifier.
        scores : CArray
            Predictions scores on surrogate data of the solver classifier.

        Returns
        -------
        x : CArray
            Surrogate data point with minimum value of objective function.

        """
        f_objs = self._objective_function_pred_scores(y_pred, scores)
        k = f_objs.argmin()
        return self._surrogate_data.X[k, :].ravel()

    def _set_alternative_init(self):
        """Set the alternative init point."""
        self.logger.info("Computing an alternative init point...")

        # Compute predictions on surrogate data if necessary
        if self._surrogate_labels is None or self._surrogate_scores is None:
            self._set_solver_surrogate_predictions()

        y_pred = self._surrogate_labels
        scores = self._surrogate_scores

        # for targeted evasion, this does not depend on the data label y0
        if self.y_target is not None:
            self._xk = self._get_point_with_min_f_obj(
                y_pred, scores.deepcopy())
            return

        # for indiscriminate evasion, this depends on y0
        # so, we compute xk for all classes
        self._xk = CArray.zeros(shape=(self.surrogate_data.num_classes,
                                       self.surrogate_data.num_features),
                                sparse=self.surrogate_data.issparse,
                                dtype=self.surrogate_data.X.dtype)
        y0 = self._y0  # Backup last y0
        for i in range(self.surrogate_data.num_classes):
            self._y0 = i
            self._xk[i, :] = self._get_point_with_min_f_obj(
                y_pred, scores.deepcopy())
        self._y0 = y0  # Restore last y0

    def _clear_solver_surrogate_predictions(self):
        """Reset the predictions on surrogate data using solver classifier."""
        super(CAttackEvasion, self)._clear_solver_surrogate_predictions()
        # After resetting predictions on surr data,
        # also reset the alternative init point
        self._xk = None

    ###########################################################################
    #                              PUBLIC METHODS
    ###########################################################################

    def _run(self, x0, y0, x_init=None, double_init=False):
        """Perform evasion for a given dmax on a single pattern.

        It solves:
            min_x g(x),
            s.t. c(x,x0) <= dmax

        Parameters
        ----------
        x0 : CArray
            Initial sample.
        y0 : int or CArray
            The true label of x0.
        x_init : CArray or None, optional
            Initialization point. If None, it is set to x0.
        double_init : bool, optional
            Whether to use or not double init for non-linear classifiers.
            Default True.

        Returns
        -------
        x_opt : CArray
            Evasion sample
        f_opt : float
            Value of objective function on x_opt (from surrogate learner).

        Notes
        -----
        Internally, this class stores the values of
         the objective function and sequence of attack points (if enabled).

        """
        # x0 must 2-D, y0 scalar if a CArray of size 1
        x0 = x0.atleast_2d()
        y0 = y0.item() if isinstance(y0, CArray) else y0

        # if data can not be modified by the attacker, exit
        if not self.is_attack_class(y0):
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
        if self._classifier.is_linear() or self.dmax == 0 or \
                double_init is False:
            return self._x_opt, self._f_opt

        # value of objective function at x_opt
        f_obj = self._solver.f_opt

        # otherwise, try to improve evasion sample
        # we run an evasion attempt using (as the init sample)
        # the sample xk with the minimum objective function from surrogate data
        if self._xk is None:
            # Choose the alternative init point if not already done
            self._set_alternative_init()

        # xk depends on whether evasion is targeted/indiscriminate
        xk = self._xk if self.y_target is not None else self._xk[self._y0, :]

        # if the projection of xk improves objective, try to restart
        xk_proj = self._solver._constr.projection(xk)

        # TODO: this has to be fixed
        # discretize x_min_proj on grid
        # xk_proj = (xk_proj / self._solver.eta).round() * self._solver.eta

        # double initialization
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
