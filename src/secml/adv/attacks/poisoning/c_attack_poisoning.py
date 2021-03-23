"""
.. module:: CAttackPoisoning
   :synopsis: Interface for poisoning attacks

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
import warnings
from abc import ABCMeta, abstractmethod

from secml.adv.attacks import CAttack, CAttackMixin
from secml.optim.optimizers import COptimizer
from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers.loss import CLoss
from secml.ml.peval.metrics import CMetric
from secml.optim.constraints import CConstraint
from secml.optim.function import CFunction


class CAttackPoisoning(CAttackMixin, metaclass=ABCMeta):
    """Interface for poisoning attacks.

    Parameters
    ----------
    classifier : CClassifier
        Target classifier.
    training_data : CDataset
        Dataset on which the the classifier has been trained on.
    val : CDataset
        Validation set.
    distance : {'l1' or 'l2'}, optional
        Norm to use for computing the distance of the adversarial example
        from the original sample. Default 'l2'.
    dmax : scalar, optional
        Maximum value of the perturbation. Default 1.
    lb, ub : int or CArray, optional
        Lower/Upper bounds. If int, the same bound will be applied to all
        the features. If CArray, a different bound can be specified for each
        feature. Default `lb = 0`, `ub = 1`.
    y_target : int or None, optional
        If None an error-generic attack will be performed, else a
        error-specific attack to have the samples misclassified as
        belonging to the `y_target` class.
    solver_type : str or None, optional
        Identifier of the solver to be used. Default 'pgd-ls'.
    solver_params : dict or None, optional
        Parameters for the solver. Default None, meaning that default
        parameters will be used.
    init_type : {'random', 'loss_based'}, optional
        Strategy used to chose the initial random samples. Default 'random'.
    random_seed : int or None, optional
        If int, random_state is the seed used by the random number generator.
        If None, no fixed seed will be set.

    """
    __super__ = 'CAttackPoisoning'

    def __init__(self, classifier,
                 training_data,
                 val,
                 distance='l2',
                 dmax=0,
                 lb=0,
                 ub=1,
                 y_target=None,
                 solver_type='pgd-ls',
                 solver_params=None,
                 init_type='random',
                 random_seed=None):

        super(CAttackPoisoning, self).__init__(
            classifier=classifier,
            distance=distance,
            dmax=dmax,
            lb=lb,
            ub=ub,
            solver_type=solver_type,
            solver_params=solver_params)

        # fixme: validation loss should be optional and passed from outside
        if classifier.class_type == 'svm':
            loss_name = 'hinge'
        elif classifier.class_type == 'logistic':
            loss_name = 'log'
        elif classifier.class_type == 'ridge':
            loss_name = 'square'
        else:
            raise NotImplementedError("We cannot poisoning that classifier")

        self._attacker_loss = CLoss.create(
            loss_name)

        self._init_loss = self._attacker_loss

        self.y_target = y_target

        # hashing xc to avoid re-training clf when xc does not change
        self._xc_hash = None

        self._x0 = None  # set the initial poisoning sample feature
        self._xc = None  # set of poisoning points along with their labels yc
        self._yc = None
        self._idx = None  # index of the current point to be optimized
        self._training_data = None  # training set used to learn classifier
        self._n_points = None  # FIXME: INIT PARAM?

        # READ/WRITE
        self.val = val  # this is for validation set
        self.training_data = training_data
        self.random_seed = random_seed

        self.init_type = init_type

        self.eta = solver_params['eta']

        # this is used to speed up some poisoning algorithms by re-using
        # the solution obtained at a previous step of the optimization
        self._warm_start = None

    @property
    def y_target(self):
        return self._y_target

    @y_target.setter
    def y_target(self, value):
        self._y_target = value

    ###########################################################################
    #                          READ-WRITE ATTRIBUTES
    ###########################################################################

    @property
    def val(self):
        """Returns the attacker's validation data"""
        return self._val

    @val.setter
    def val(self, value):
        """Sets the attacker's validation data"""
        if value is None:
            self._val = None
            return
        if not isinstance(value, CDataset):
            raise TypeError('val should be a CDataset!')
        self._val = value

    @property
    def training_data(self):
        """Returns the training set used to learn the targeted classifier"""
        return self._training_data

    @training_data.setter
    def training_data(self, value):
        """Sets the training set used to learn the targeted classifier"""
        # mandatory parameter, we raise error also if value is None
        if not isinstance(value, CDataset):
            raise TypeError('training_data should be a CDataset!')
        self._training_data = value

    @property
    def random_seed(self):
        """Returns the attacker's validation data"""
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value):
        """Sets the attacker's validation data"""
        self._random_seed = value

    @property
    def n_points(self):
        """Returns the number of poisoning points."""
        return self._n_points

    @n_points.setter
    def n_points(self, value):
        """Sets the number of poisoning points."""
        if value is None:
            self._n_points = None
            return
        self._n_points = int(value)

    @property
    def x0(self):
        """Returns the attacker's initial sample features"""
        return self._x0

    @x0.setter
    def x0(self, value):
        """Set the attacker's initial sample features"""
        self._x0 = value

    @property
    def xc(self):
        """Returns the attacker's sample features"""
        return self._xc

    @xc.setter
    def xc(self, value):
        """Set the attacker's sample features"""
        self._xc = value

    @property
    def yc(self):
        """Returns the attacker's sample label"""
        return self._yc

    @yc.setter
    def yc(self, value):
        """Set the attacker's sample label"""
        self._yc = value

    ###########################################################################
    #                              PRIVATE METHODS
    ###########################################################################

    def _constraint_creation(self):

        # only feature increments or decrements are allowed
        lb = self._x0 if self.lb == 'x0' else self.lb
        ub = self._x0 if self.ub == 'x0' else self.ub
        bounds = CConstraint.create('box', lb=lb, ub=ub)

        constr = CConstraint.create(self.distance, center=0, radius=1e12)

        return bounds, constr

    def _init_solver(self):
        """Create solver instance."""

        if self.classifier is None:
            raise ValueError('Solver not set properly!')

        # map attributes to fun, constr, box
        fun = CFunction(fun=self.objective_function,
                        gradient=self.objective_function_gradient,
                        n_dim=self._classifier.n_features)

        bounds, constr = self._constraint_creation()

        self._solver = COptimizer.create(
            self._solver_type,
            fun=fun, constr=constr,
            bounds=bounds,
            **self.solver_params)

        self._solver.verbose = 0
        self._warm_start = None

    def _rnd_init_poisoning_points(
            self, n_points=None, init_from_val=False, val=None):
        """Returns a random set of poisoning points randomly with
        flipped labels."""
        if init_from_val:
            if val:
                init_dataset = val
            else:
                init_dataset = self.val
        else:
            init_dataset = self.training_data

        if (self._n_points is None or self._n_points == 0) and (
                n_points is None or n_points == 0):
            raise ValueError("Number of poisoning points (n_points) not set!")

        if n_points is None:
            n_points = self.n_points

        idx = CArray.randsample(init_dataset.num_samples, n_points,
                                random_state=self.random_seed)

        xc = init_dataset.X[idx, :].deepcopy()

        # if the attack is in a continuous space we add a
        # little perturbation to the initial poisoning point
        random_noise = CArray.rand(shape=xc.shape,
                                   random_state=self.random_seed)
        xc += 1e-3 * (2 * random_noise - 1)
        yc = CArray(init_dataset.Y[idx]).deepcopy()  # true labels

        # randomly pick yc from a different class
        for i in range(yc.size):
            labels = CArray.randsample(init_dataset.num_classes, 2,
                                       random_state=self.random_seed)
            if yc[i] == labels[0]:
                yc[i] = labels[1]
            else:
                yc[i] = labels[0]

        return xc, yc

    def _update_poisoned_clf(self, clf=None, tr=None,
                             train_normalizer=False):
        """
        Trains classifier on D (original training data) plus {x,y} (new point).

        Parameters
        ----------
        x: feature vector of new training point
        y: true label of new training point

        Returns
        -------
        clf: trained classifier on D and {x,y}

        """

        #  xc hashing is only valid if clf and tr do not change
        #  (when calling update_poisoned_clf() without parameters)
        xc_hash_is_valid = False
        if clf is None and tr is None:
            xc_hash_is_valid = True

        if clf is None:
            clf = self.classifier

        if tr is None:
            tr = self.training_data

        tr = tr.append(CDataset(self._xc, self._yc))

        xc_hash = self._xc.sha1()

        if self._xc_hash is None or self._xc_hash != xc_hash:
            # xc set has changed, retrain clf
            # hash is stored only if update_poisoned_clf() is called w/out pars
            self._xc_hash = xc_hash if xc_hash_is_valid else None
            self._poisoned_clf = clf.deepcopy()

            # we assume that normalizer is not changing w.r.t xc!
            # so we avoid re-training the normalizer on dataset including xc

            if self.classifier.preprocess is not None:
                self._poisoned_clf.retrain_normalizer = train_normalizer

            self._poisoned_clf.fit(tr.X, tr.Y)

        return self._poisoned_clf, tr

    ###########################################################################
    #                  OBJECTIVE FUNCTION & GRAD COMPUTATION
    ###########################################################################

    def objective_function(self, xc, acc=False):
        """
        Parameters
        ----------
        xc: poisoning point

        Returns
        -------
        f_obj: values of objective function (average hinge loss) at x
        """

        # index of poisoning point within xc.
        # This will be replaced by the input parameter xc
        if self._idx is None:
            idx = 0
        else:
            idx = self._idx

        xc = CArray(xc).atleast_2d()

        n_samples = xc.shape[0]
        if n_samples > 1:
            raise TypeError("xc is not a single sample!")

        self._xc[idx, :] = xc
        clf, tr = self._update_poisoned_clf()

        y_pred, score = clf.predict(self.val.X, return_decision_function=True)

        # targeted attacks
        y_ts = CArray(self._y_target).repeat(score.shape[0]) \
            if self._y_target is not None else self.val.Y

        # TODO: binary loss check
        if self._attacker_loss.class_type != 'softmax':
            score = CArray(score[:, 1].ravel())

        if acc is True:
            error = CArray(y_ts != y_pred).ravel()  # compute test error
        else:
            error = self._attacker_loss.loss(y_ts, score)
        obj = error.mean()

        return obj

    def objective_function_gradient(self, xc, normalization=True):
        """
        Compute the loss derivative wrt the attack sample xc

        The derivative is decomposed as:

        dl / x = sum^n_c=1 ( dl / df_c * df_c / x )
        """

        xc = xc.atleast_2d()
        n_samples = xc.shape[0]

        if n_samples > 1:
            raise TypeError("x is not a single sample!")

        # index of poisoning point within xc.
        # This will be replaced by the input parameter xc
        if self._idx is None:
            idx = 0
        else:
            idx = self._idx

        self._xc[idx, :] = xc
        clf, tr = self._update_poisoned_clf()

        # computing gradient of loss(y, f(x)) w.r.t. f
        _, score = clf.predict(self.val.X, return_decision_function=True)

        y_ts = CArray(self._y_target).repeat(score.shape[0]) \
            if self._y_target is not None else self.val.Y

        grad = CArray.zeros((xc.size,))

        if clf.n_classes <= 2:
            loss_grad = self._attacker_loss.dloss(
                y_ts, CArray(score[:, 1]).ravel())
            grad = self._gradient_fk_xc(
                self._xc[idx, :], self._yc[idx], clf, loss_grad, tr)
        else:
            # compute the gradient as a sum of the gradient for each class
            for c in range(clf.n_classes):
                loss_grad = self._attacker_loss.dloss(y_ts, score, c=c)

                grad += self._gradient_fk_xc(self._xc[idx, :], self._yc[idx],
                                             clf, loss_grad, tr, c)

        if normalization:
            norm = grad.norm()
            return grad / norm if norm > 0 else grad
        else:
            return grad

    @abstractmethod
    def _gradient_fk_xc(self, xc, yc, clf, loss_grad, tr, k=None):
        """
        Derivative of the classifier's discriminant function f_k(x)
        computed on a set of points x w.r.t. a single poisoning point xc

        This is a classifier-specific implementation, so we delegate its
        implementation to inherited classes.
        """
        pass

    ###########################################################################
    #         POISONING INTERNAL ROUTINE ON SINGLE DATA POINT (PRIVATE)
    ###########################################################################

    def _run(self, xc, yc, idx=0):
        """Single point poisoning.

        Here xc can be a *set* of points, in which case idx specifies which
        point should be manipulated by the poisoning attack.

        """
        xc = CArray(xc.deepcopy()).atleast_2d()

        self._yc = yc
        self._xc = xc
        self._idx = idx  # point to be optimized within xc

        self._x0 = self._xc[idx, :].ravel()

        self._init_solver()

        if self.y_target is None:  # indiscriminate attack
            x = self._solver.maximize(self._x0)
        else:  # targeted attack
            x = self._solver.minimize(self._x0)

        self._solution_from_solver()

        return x

    ###########################################################################
    #                              PUBLIC METHODS
    ###########################################################################

    def run(self, x, y, ds_init=None, max_iter=1):
        """Runs poisoning on multiple points.

        It reads n_points (previously set), initializes xc, yc at random,
        and then optimizes the poisoning points xc.

        Parameters
        ----------
        x : CArray
            Validation set for evaluating classifier performance.
            Note that this is not the validation data used by the attacker,
            which should be passed instead to `CAttackPoisoning` init.
        y : CArray
            Corresponding true labels for samples in `x`.
        ds_init : CDataset or None, optional.
            Dataset for warm start.
        max_iter : int, optional
            Number of iterations to re-optimize poisoning data. Default 1.

        Returns
        -------
        y_pred : predicted labels for all val samples by targeted classifier
        scores : scores for all val samples by targeted classifier
        adv_xc : manipulated poisoning points xc (for subsequents warm starts)
        f_opt : final value of the objective function

        """
        if self._n_points is None or self._n_points == 0:
            # evaluate performance on x,y
            y_pred, scores = self._classifier.predict(
                x, return_decision_function=True)
            return y_pred, scores, ds_init, 0

        # n_points > 0
        if self.init_type == 'random':
            # randomly sample xc and yc
            xc, yc = self._rnd_init_poisoning_points()
        elif self.init_type == 'loss_based':
            xc, yc = self._loss_based_init_poisoning_points()
        else:
            raise NotImplementedError(
                "Unknown poisoning point initialization strategy.")

        # re-set previously-optimized points if passed as input
        if ds_init is not None:
            xc[0:ds_init.num_samples, :] = ds_init.X
            yc[0:ds_init.num_samples] = ds_init.Y

        delta = 1.0
        k = 0

        # max_iter ignored for single-point attacks
        if self.n_points == 1:
            max_iter = 1

        metric = CMetric.create('accuracy')

        while delta > 0 and k < max_iter:

            self.logger.info(
                "Iter on all the poisoning samples: {:}".format(k))

            xc_prv = xc.deepcopy()
            for i in range(self._n_points):
                # this is to optimize the last points first
                # (and then re-optimize the first ones)
                idx = self.n_points - i - 1
                xc[idx, :] = self._run(xc, yc, idx=idx)
                # optimizing poisoning point 0
                self.logger.info(
                    "poisoning point {:} optim fopt: {:}".format(
                        i, self._f_opt))

                y_pred, scores = self._poisoned_clf.predict(
                    x, return_decision_function=True)
                acc = metric.performance_score(y_true=y, y_pred=y_pred)
                self.logger.info("Poisoned classifier accuracy "
                                 "on test data {:}".format(acc))

            delta = (xc_prv - xc).norm_2d()
            self.logger.info(
                "Optimization with n points: " + str(self._n_points) +
                " iter: " + str(k) + ", delta: " +
                str(delta) + ", fopt: " + str(self._f_opt))
            k += 1

        # re-train the targeted classifier (copied) on poisoned data
        # to evaluate attack effectiveness on targeted classifier
        clf, tr = self._update_poisoned_clf(clf=self._classifier,
                                            tr=self._training_data,
                                            train_normalizer=False)
        # fixme: rechange train_normalizer=True

        y_pred, scores = clf.predict(x, return_decision_function=True)
        acc = metric.performance_score(y_true=y, y_pred=y_pred)
        self.logger.info(
            "Original classifier accuracy on test data {:}".format(acc))

        return y_pred, scores, CDataset(xc, yc), self._f_opt

    def _loss_based_init_poisoning_points(self, n_points=None):
        """
        """
        raise NotImplementedError

    def _compute_grad_inv(self, G, H, grad_loss_params):

        from scipy import linalg
        det = linalg.det(H.tondarray())
        if abs(det) < 1e-6:
            H_inv = CArray(linalg.pinv2(H.tondarray()))
        else:
            H_inv = CArray(linalg.inv(H.tondarray()))
        grad_mat = - CArray(G.dot(H_inv))  # d * (d + 1)

        self._d_params_xc = grad_mat

        gt = grad_mat.dot(grad_loss_params)
        return gt.ravel()

    def _compute_grad_solve(self, G, H, grad_loss_params, sym_pos=True):

        from scipy import linalg
        v = linalg.solve(
            H.tondarray(), grad_loss_params.tondarray(), sym_pos=sym_pos)
        v = CArray(v)
        gt = -G.dot(v)
        return gt.ravel()

    def _compute_grad_solve_iterative(self, G, H, grad_loss_params, tol=1e-6):
        from scipy.sparse import linalg

        if self._warm_start is None:
            v, convergence = linalg.cg(
                H.tondarray(), grad_loss_params.tondarray(), tol=tol)
        else:
            v, convergence = linalg.cg(
                H.tondarray(), grad_loss_params.tondarray(), tol=tol,
                x0=self._warm_start.tondarray())

        if convergence != 0:
            warnings.warn('Convergence of poisoning algorithm not reached!')

        v = CArray(v.ravel())

        # store v to be used as warm start at the next iteration
        self._warm_start = v

        gt = -G.dot(v.T)
        return gt.ravel()
