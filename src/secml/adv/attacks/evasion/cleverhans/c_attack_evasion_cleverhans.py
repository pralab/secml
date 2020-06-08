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

from secml.adv.attacks import CAttack
from secml.adv.attacks.evasion import CAttackEvasion
from secml.adv.attacks.evasion.cleverhans.c_attack_evasion_cleverhans_losses \
    import CAttackEvasionCleverhansLossesMixin
from secml.array import CArray
from secml.core import CCreator
from secml.core.constants import nan
from secml.core.exceptions import NotFittedError
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
        Target classifier (trained).
    y_target : int or None, optional
        If None an indiscriminate attack will be performed, else a
        targeted attack to have the samples misclassified as
        belonging to the y_target class.
    clvh_attack_class:
        The CleverHans class that implement the attack
    store_var_list: list
        list of variables to store from the graph during attack
        run. The variables will be stored as key-value dictionary
        and can be retrieved through the property `stored_vars`.

    **kwargs
        Any other parameter for the cleverhans attack.

    Notes
    -----
    The current Tensorflow default graph will be used.

    """
    __class_type = 'e-cleverhans'

    def __init__(self, classifier, y_target=None,
                 clvh_attack_class=CarliniWagnerL2,
                 store_var_list=None, **kwargs):

        self._tfsess = tf.compat.v1.Session()

        self._eps_0 = False

        # store the cleverhans attack parameters
        self.attack_params = kwargs

        # Check if the cleverhans attack is supported
        if clvh_attack_class not in SUPPORTED_ATTACKS:
            raise ValueError("This cleverhans attack is not supported yet!")

        self._clvrh_attack_class = clvh_attack_class

        self._clvrh_clf = None

        self._last_f_eval = 0
        self._last_grad_eval = 0

        if store_var_list is not None:
            # first, check if the user has set stored variables
            self._stored_vars = {k: [] for k in store_var_list}
        elif any([self._clvrh_attack_class == CarliniWagnerL2,
                  self._clvrh_attack_class == ElasticNetMethod, ]):
            # store `const` by default for these attacks as it
            # is needed in the `objective_function` computation
            self._stored_vars = {'const': []}
        else:
            self._stored_vars = None

        super(CAttackEvasionCleverhans, self).__init__(
            classifier=classifier,
            y_target=y_target)

        self._n_classes = self._classifier.n_classes
        self._n_feats = self._classifier.n_features

        self._initialize_tf_ops()

    def set(self, param_name, param_value, copy=False):

        # we need the possibility of running the attack for eps==0,
        # this is not allowed in standard cleverhans
        if 'eps' in param_name:
            if param_value == 0:
                param_value = 1
                self._eps_0 = True
            else:
                self._eps_0 = False

        if param_name.startswith('attack_params'):
            super(CAttackEvasionCleverhans, self).set(param_name, param_value,
                                                      copy)

        # re-initialize the Tensorflow operations
        self._initialize_tf_ops()

    def run(self, x, y, ds_init=None):
        # override run for applying storage of internally
        # optimized variables
        if self._stored_vars is not None:
            for key in self._stored_vars:
                self._stored_vars[key] = []
        return super(CAttackEvasionCleverhans, self).run(
            x, y, ds_init=ds_init)

    ###########################################################################
    #                           READ-ONLY ATTRIBUTES
    ###########################################################################

    @property
    def f_eval(self):
        if self._clvrh_clf:
            return self._last_f_eval
        else:
            raise ValueError("Attack not performed yet!")

    @property
    def grad_eval(self):
        if self._clvrh_clf:
            return self._last_grad_eval
        else:
            raise ValueError("Attack not performed yet!")

    @property
    def stored_vars(self):
        """Variables extracted from the graph during execution of the attack.

        """
        return self._stored_vars

    @property
    def attack_params(self):
        """Object containing all Cleverhans parameters

        """
        return self._attack_params

    @attack_params.setter
    def attack_params(self, value):
        """Object containing all Cleverhans parameters

        """
        self._attack_params = _CClvrh_params(value)

    ###########################################################################
    #                              PRIVATE METHODS
    ###########################################################################

    def objective_function(self, x):
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
        elif self._clvrh_attack_class == ElasticNetMethod:
            return self._objective_function_elastic_net(x)
        elif self._clvrh_attack_class == SPSA:
            return self._objective_function_SPSA(x)
        elif self._clvrh_attack_class in [
            FastGradientMethod, ProjectedGradientDescent, LBFGS,
            MomentumIterativeMethod, MadryEtAl, BasicIterativeMethod]:
            return self._objective_function_cross_entropy(x)
        else:
            raise NotImplementedError

    def objective_function_gradient(self, x):
        """Gradient of the objective function."""
        raise NotImplementedError

    def _create_tf_operations(self):
        """
        Call `generate` from the Cleverhans attack to
        construct the Tensorflow operation needed to perform the attack

        """
        if self.y_target is None:
            if 'y' in self._clvrh_attack.feedable_kwargs:
                self._adv_x_T = self._clvrh_attack.generate(
                    self._initial_x_P, y=self._y_P,
                    **self.attack_params.__dict__)
            else:  # 'y' not required by attack
                self._adv_x_T = self._clvrh_attack.generate(
                    self._initial_x_P, **self.attack_params.__dict__)
        else:
            if 'y_target' not in self._clvrh_attack.feedable_kwargs:
                raise RuntimeError(
                    "cannot perform a targeted {:} attack".format(
                        self._clvrh_attack.__class__.__name__))
            self._adv_x_T = self._clvrh_attack.generate(
                self._initial_x_P, y_target=self._y_P,
                **self._attack_params.__dict__)

    def _initialize_tf_ops(self):

        # create the cleverhans attack object
        tf.reset_default_graph()
        self._tfsess.close()
        session_conf = tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=-1,  # Perform in caller's thread
            use_per_session_threads=False  # Per-session thread pools
        )
        self._tfsess = tf.compat.v1.Session(config=session_conf)

        # wrap the surrogate classifier into a cleverhans classifier
        self._clvrh_clf = _CModelCleverhans(
            self.classifier, out_dims=self._n_classes)

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

        # call the function of the cleverhans attack called `generate` that
        # constucts the Tensorflow operation needed to perform the attack
        self._create_tf_operations()

    def _define_warning_filter(self):
        # We filter few warnings raised by numpy, caused by cleverhans

        self.logger.filterwarnings(
            "ignore", category=RuntimeWarning,
            message="invalid value encountered in double_scalars*"
        )
        self.logger.filterwarnings(
            "ignore", category=RuntimeWarning,
            message="Mean of empty slice*"
        )

    def _create_one_hot_y(self):
        """
        Cleverhans attacks need to receive y as a one hot vector.
        y is equal to the y target if y_target is present, otherwhise is
        equal to the true class of the attack sample.

        """
        one_hot_y = CArray.zeros(shape=(1, self._n_classes),
                                 dtype=np.float32)

        if self.y_target is not None:
            one_hot_y[0, self.y_target] = 1
        else:  # indiscriminate attack
            one_hot_y[0, self._y0.item()] = 1

        return one_hot_y

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
        self._clvrh_clf.reset_eval()

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

        if self._eps_0 is True:
            return x0, nan

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
            self._define_warning_filter()

            # Cleverhans attacks need to receive y as a one hot vector.
            # y is equal to the y target if y_target is present, otherwhise is
            # equal to the true class of the attack sample.
            one_hot_y = self._create_one_hot_y()

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

        if self._stored_vars is not None:
            for key in self._stored_vars:
                self._stored_vars[key].append(self._get_variable_value(key))

        self._last_f_eval = self._clvrh_clf.f_eval
        self._last_grad_eval = self._clvrh_clf.grad_eval

        return self._x_opt, nan  # TODO: return value of objective_fun(x_opt)

    def _get_variable_value(self, var_name):
        const = self._clvrh_clf.get_variable_value(var_name)
        const_value = self._tfsess.run(const)
        return const_value


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
    The Tensorflow model will be added to the current Tensorflow default graph.

    """

    @property
    def f_eval(self):
        return self._fun.n_fun_eval

    @property
    def grad_eval(self):
        return self._fun.n_grad_eval

    def reset_eval(self):
        """Reset the number of evaluations."""
        self._fun.reset_eval()

    def _decision_function(self, x):
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
        return self._clf.forward(x, caching=True)  # TODO: caching required?

    def __init__(self, clf, out_dims=None):

        self._clf = clf

        if isinstance(clf, CClassifierReject):
            raise ValueError("classifier with reject cannot be "
                             "converted to a tensorflow model")

        if not clf.is_fitted():
            raise NotFittedError("The classifier should be already trained!")

        self._out_dims = out_dims

        # classifier output tensor name. Either "probs" or "logits".
        self._output_layer = 'logits'

        # Given a trained CClassifier, creates a tensorflow node for the
        # network output and one for its gradient
        self._fun = CFunction(fun=self._decision_function,
                              gradient=clf.gradient)
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

    def get_variable_value(self, variable_name):
        return tf.get_default_graph().get_tensor_by_name(
            "{:}:0".format(variable_name))


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

        grad_f_x = self.fun.gradient
        grads = grad_f_x(x_carray, w=grads_in_np).atleast_2d()

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
    Given a function that returns as output a numpy array, and optionally a
    function that computes its gradient, this function returns a pyfunction.
    A pyfunction wraps a function that works with numpy arrays as an operation
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


class _CClvrh_params(CCreator):
    def __init__(self, param_dict):
        # create a property for each dictionary key
        self.__dict__ = param_dict
