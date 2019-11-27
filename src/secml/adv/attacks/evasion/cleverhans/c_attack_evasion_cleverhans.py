"""
.. module:: CAttackEvasionCleverhans
    :synopsis: Performs one of the Cleverhans Evasion attacks
                against a classifier.

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""
import numpy as np
import tensorflow as tf
from cleverhans.attacks import \
    FastGradientMethod, CarliniWagnerL2, ElasticNetMethod, SPSA, LBFGS, \
    ProjectedGradientDescent, SaliencyMapMethod, MomentumIterativeMethod, \
    MadryEtAl, BasicIterativeMethod, DeepFool
from cleverhans.model import Model

from secml.adv.attacks.evasion.cleverhans.c_attack_evasion_cleverhans_losses import \
    CAttackEvasionCleverhansLossesMixin
from secml.core.exceptions import NotFittedError

from secml.adv.attacks import CAttack
from secml.adv.attacks.evasion import CAttackEvasion
from secml.array import CArray
from secml.core.constants import nan
from secml.ml.classifiers.reject import CClassifierReject
from secml.optim.function import CFunction

SUPPORTED_ATTACKS = [
    FastGradientMethod, CarliniWagnerL2, ElasticNetMethod, SPSA, LBFGS,
    ProjectedGradientDescent, SaliencyMapMethod, MomentumIterativeMethod,
    MadryEtAl, BasicIterativeMethod, DeepFool
]


class CAttackEvasionCleverhans(CAttackEvasion,
                               CAttackEvasionCleverhansLossesMixin):
    """This class is a wrapper of the attacks implemented in the Cleverhans
    library.
    
    Credits: https://github.com/tensorflow/cleverhans.

    Parameters
    ----------
    classifier : CClassifier
        Target classifier on which the efficacy of the computed attack
        points is evaluates
    n_feats : int
        Number of features of the dataset used to train the classifiers.
    surrogate_classifier : CClassifier
        Surrogate classifier against which the attack is computed.
        This is assumed to be already trained on surrogate_data.
    surrogate_data: CDataset
        Used to train the surrogate classifier.
    y_target : int or None, optional
            If None an indiscriminate attack will be performed, else a
            targeted attack to have the samples misclassified as
            belonging to the y_target class.
    clvh_attack_class
        The CleverHans class that implement the attack
    **kwargs
        Any other parameter for the cleverhans attack.

    Notes
    -----
    The current Tensorflow default graph will be used.

    """
    class_type = 'e-cleverhans'

    def __init__(self, classifier, surrogate_classifier,
                 n_feats, n_classes, surrogate_data=None, y_target=None,
                 clvh_attack_class=CarliniWagnerL2, **kwargs):

        self._tfsess = tf.compat.v1.Session()

        # store the cleverhans attack parameters
        self._clvrh_params = kwargs

        # Check if the cleverhans attack is supported
        if clvh_attack_class not in SUPPORTED_ATTACKS:
            raise ValueError("This cleverhans attack is not supported yet!")

        self._clvrh_attack_class = clvh_attack_class

        # store the number of features
        self._n_feats = n_feats
        # store the number of dataset classes
        self._n_classes = n_classes

        self._clvrh_clf = None

        CAttackEvasion.__init__(self, classifier=classifier,
                                surrogate_classifier=surrogate_classifier,
                                surrogate_data=surrogate_data,
                                y_target=y_target)

    ###########################################################################
    #                           READ-ONLY ATTRIBUTES
    ###########################################################################

    @property
    def f_eval(self):
        if self._clvrh_clf:
            return self._clvrh_clf.f_eval
        else:
            return 0

    @property
    def grad_eval(self):
        if self._clvrh_clf:
            return self._clvrh_clf.grad_eval
        else:
            return 0

    ###########################################################################
    #                              PRIVATE METHODS
    ###########################################################################

    def _objective_function(self, x):
        """Objective function.

        Parameters
        ----------
        x : CArray or CDataset

        Returns
        -------
        f_obj : float or CArray of floats

        """
        if self._clvrh_attack_class == CarliniWagnerL2:
            return self._objective_function_cw(x)
        elif self._clvrh_attack_class in [
                FastGradientMethod, ProjectedGradientDescent, LBFGS,
                MomentumIterativeMethod, MadryEtAl, BasicIterativeMethod]:
            return self._objective_function_cross_entropy(x)
        else:
            raise NotImplementedError

    def _objective_function_gradient(self, x):
        """Gradient of the objective function."""
        raise NotImplementedError

    def _set_solver_classifier(self):
        """This function set the surrogate classifier,
        if differentiable; otherwise, it learns a smooth approximation for
        the nondiff. (surrogate) classifier (e.g., decision tree)
        using an SVM with the RBF kernel."""

        # update the surrogate classifier
        # we skip the function provided by the superclass as we do not need
        # to set xk and we call directly the one of CAttack that instead
        # learn a differentiable classifier
        CAttack._set_solver_classifier(self)

        # create the cleverhans attack object
        self._tfsess.close()
        session_conf = tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=-1,  # Perform in caller's thread
            use_per_session_threads=False    # Per-session thread pools
        )
        self._tfsess = tf.compat.v1.Session(config=session_conf)

        # wrap the surrogate classifier into a cleverhans classifier
        self._clvrh_clf = _CModelCleverhans(
            self._surrogate_classifier, out_dims=self._n_classes)
        # create an instance of the chosen cleverhans attack
        self._clvrh_attack = self._clvrh_attack_class(
            self._clvrh_clf, sess=self._tfsess)

        # create the placeholder to feed into the attack the initial evasion
        # samples
        self._initial_x_P = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self._n_feats))

        # placeholder used to feed the true or the target label (it is a
        # one-hot encoded vector)
        self._y_P = tf.compat.v1.placeholder(
            tf.float32, shape=(1, self._n_classes))

        # create the tf operations to generate the attack
        if self.y_target is None:
            if 'y' in self._clvrh_attack.feedable_kwargs:
                self._adv_x_T = self._clvrh_attack.generate(
                    self._initial_x_P, y=self._y_P, **self._clvrh_params)
            else:  # 'y' not required by attack
                self._adv_x_T = self._clvrh_attack.generate(
                    self._initial_x_P, **self._clvrh_params)
        else:
            if 'y_target' not in self._clvrh_attack.feedable_kwargs:
                raise RuntimeError(
                    "cannot perform a targeted {:} attack".format(
                        self._clvrh_attack.__class__.__name__))
            self._adv_x_T = self._clvrh_attack.generate(
                self._initial_x_P, y_target=self._y_P, **self._clvrh_params)

    def _run(self, x0, y0, x_init=None):
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
        x = self._x0.atleast_2d().tondarray().astype(np.float32)

        # initialize caching of the attack path
        self._clvrh_clf.reset_caching(x0)

        # create a one-hot-encoded vector to feed the true or
        # the y_target label

        one_hot_y = CArray.zeros(shape=(1, self._n_classes),
                                 dtype=np.float32)

        if self.y_target is not None:
            one_hot_y[0, self.y_target] = 1
        else:  # indiscriminate attack
            one_hot_y[0, self._y0.item()] = 1

        with self.logger.catch_warnings():
            # We filter few warnings raised by numpy, caused by cleverhans

            self.logger.filterwarnings(
                "ignore", category=RuntimeWarning,
                message="invalid value encountered in double_scalars*"
            )
            self.logger.filterwarnings(
                "ignore", category=RuntimeWarning,
                message="Mean of empty slice*"
            )

            self._x_opt = self._tfsess.run(
                self._adv_x_T, feed_dict={self._initial_x_P: x,
                                          self._y_P: one_hot_y.tondarray()})

        self._x_opt = CArray(self._x_opt)
        self._x_seq = self._clvrh_clf._x_seq
        if (self.x_opt - self._x_seq[0, :]).norm_2d() > 1e-6:
            self._x_seq = self._x_seq.append(self._x_opt, axis=0)
        else:
            # some attack returns at the initial point if
            # condition is not met (e.g. CWL2)
            self._x_opt = self._x_seq[-1, :]
        return self._x_opt, nan


class _CModelCleverhans(Model):
    """Receive our library classifier and convert it into a cleverhans model.

    Parameters
    ----------
    clf : CClassifier
        SecML classifier, should be already trained.
    out_dims : int or None
        The expected number of classes.

    Notes
    -----
    The Tesorflow model will be created in the current
    Tensorflow default graph.

    """

    @property
    def f_eval(self):
        return self._fun.n_grad_eval

    @property
    def grad_eval(self):
        return self._fun.n_grad_eval

    def _discriminant_function(self, x):
        """
        Wrapper of the classifier discriminant function. This is needed
        because the output of the CFunction should be either a scalar or a
        CArray whereas the predict function returns a tuple.
        """
        if hasattr(self, '_x_seq') and self._x_seq is not None:
            if self._is_init is True:  # avoid storing twice the initial value
                self._is_init = False
            else:  # Cache intermediate values
                self._x_seq = self._x_seq.append(x, axis=0)
        return self._clf.predict(x, return_decision_function=True)[1]

    def __init__(self, clf, out_dims=None):

        self._clf = clf

        if isinstance(clf, CClassifierReject):
            raise ValueError("classifier with reject cannot be "
                             "converted as tensoflow model")

        if not clf.is_fitted():
            raise NotFittedError("The classifier should be already trained!")

        self._out_dims = out_dims

        # classifier output tensor name. Either "probs" or "logits".
        self._output_layer = 'logits'

        # Given a trained CClassifier, creates a tensorflow node for the
        # network output and one for its gradient
        self._fun = CFunction(fun=self._discriminant_function,
                              gradient=clf.grad_f_x)
        self._callable_fn = _CClassifierToTF(self._fun, self._out_dims)

        super(_CModelCleverhans, self).__init__(nb_classes=clf.n_classes)

    def fprop(self, x, **kwargs):
        """

        Parameters
        ----------
        x : np.ndarray
            Input samples.
        **kwargs : dict
            Any other argument for function.

        Returns
        -------
        dict

        """
        if len(kwargs) != 0:  # TODO: SUPPORT KWARGS
            raise ValueError("kwargs not supported")
        return {self._output_layer: self._callable_fn(x, **kwargs)}

    def reset_caching(self, x=None):
        """Sets the caching for storing the attack path."""
        if x is not None:
            self._x_seq = CArray(x)
            # this is used for avoiding double storage
            # of the initial path
            self._is_init = True
        else:
            self._x_seq = None


class _CClassifierToTF:
    """
    Creates a Tensorflow operation whose result are the scores produced by
    the discriminant function of a CClassifier subclass.
    Accordingly, the gradient of the discriminant function will be the gradient
    of the created Tensorflow operation.

    Parameters
    ----------
    fun : CFunction that have as function the discriminant function of the
     classifier.
    out_dims : The number of output dimensions (classes) of the model.

    Returns
    -------
    tf_model_fn
        A Tensorfow operation that maps an input (tf.Tensor) to the output
        of model discriminant function (tf.Tensor) and on which the
        tensorflow function "gradient" can be called to compute its gradient.

    """

    def __init__(self, fun, out_dims=1):
        self.fun = fun
        self.out_dims = out_dims

    def __call__(self, x_op):
        """
        Given an input, this function return a PyFunction (a Tensorflow
        operation) that compute the function "_fprop_fn" and whose
        gradient is the one give by the function "_tf_gradient_fn".

        Returns
        -------
        out: PyFunction
            tensorflow operation on which the gradient function can be
            used to have its gradient.

        """
        out = _py_func_with_gradient(self._fprop_fn, [x_op],
                                     Tout=[tf.float32],
                                     stateful=True,
                                     grad_func=self._tf_gradient_fn)[0]
        out.set_shape([None, self.out_dims])

        return out

    def _fprop_fn(self, x_np):
        """Numpy function that computes and returns the output of the model.

        Parameters
        ----------
        x_np: np.ndarray
            The input that should be classified by the model.

        Returns
        -------
        scores: np.ndarray
            The scores given as output by the classifier.

        """
        # compute the scores (the model output)
        f_x = self.fun.fun
        scores = f_x(CArray(x_np)).atleast_2d().tondarray().astype(np.float32)

        return scores

    def _np_grad_fn(self, x_np, grads_in_np=None):
        """
        Function that compute the gradient of the classifier discriminant
        function

        Parameters
        ----------
        x_np: np.ndarray
            Input samples (2D array)
        grads_in_np: np.ndarray
            Vector for which the gradient should be pre-multiplied.

        Returns
        -------
        gradient: np.ndarray
            The gradient of the classifier discriminant function.

        """
        x_carray = CArray(x_np).atleast_2d()
        grads_in_np = CArray(grads_in_np).atleast_2d()

        n_samples = x_carray.shape[0]

        if n_samples > 1:
            raise ValueError("The gradient of CCleverhansAttack can be "
                             "computed only for one sample at time")

        n_feats = x_carray.shape[1]
        n_classes = grads_in_np.shape[1]

        grads = CArray.zeros((n_samples, n_feats))

        # if grads_in_np we can speed up the computation computing just one
        # gradient
        grad_f_x = self.fun.gradient
        if grads_in_np.sum(axis=None) == 1:
            y = grads_in_np.find(grads_in_np == 1)[0]
            if grads_in_np[y] == 1:
                grads = grad_f_x(x_carray, y=y).atleast_2d()
        else:
            # otherwise we have to compute the gradient w.r.t all the classes
            for c in range(n_classes):
                cgrad = grad_f_x(x_carray[0, :], y=c)
                grads[0, :] += (cgrad * CArray(grads_in_np)[0, c])

        return grads.tondarray().astype(np.float32)

    def _tf_gradient_fn(self, op, grads_in):
        """

        Parameters
        ----------
        op : numpy function.
        grads_in : input of the gradient function.

        Returns
        -------
        gradient: PyFunction
            tensorflow operation that compute the gradient of a given
            function that works with numpy array.

        """
        pyfun = tf.compat.v1.py_func(
            self._np_grad_fn, [op.inputs[0], grads_in], Tout=[tf.float32])
        return pyfun


def _py_func_with_gradient(
        func, inp, Tout, stateful=True, pyfun_name=None, grad_func=None):
    """
    Given a function that returns as output a numpy array, and eventually a
    function that computes his gradient, this function returns a pyfunction.
    A pyfunction wrap a function that works with numpy array as an operation
    in a TensorFlow graph.

    credits: https://gist.github.com/kingspp/3ec7d9958c13b94310c1a365759aa3f4

    Parameters
    ----------
    func : Custom Function.
    inp : Function Inputs.
    Tout : Output Type of out Custom Function.
    stateful : Calculate Gradients when stateful is True.
    pyfun_name : Name of the PyFunction.
    grad_func : Custom Gradient Function.

    Returns
    ----------
    fun : Pyfunction
        Tensorflow operation that compute the given function and eventually
        its gradient.

    """
    # Generate random name in order to avoid conflicts with inbuilt names
    from random import getrandbits
    rnd_name = 'PyFuncGrad-' + '%0x' % getrandbits(30 * 4)

    # Register Tensorflow Gradient
    tf.RegisterGradient(rnd_name)(grad_func)

    # Get current graph
    g = tf.compat.v1.get_default_graph()

    # Add gradient override map
    with g.gradient_override_map(
            {"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.compat.v1.py_func(
            func, inp, Tout, stateful=stateful, name=pyfun_name)
