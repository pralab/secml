"""
.. module:: CModelCleverhans
   :synopsis: Class that, given a CClassifier, wrap it into a Cleverhans Model.
             We use this class to be able to run the cleverhans attacks on
            Cleverhans classifiers.

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from random import getrandbits
from six.moves import range

import numpy as np

import tensorflow as tf
from cleverhans.model import Model

from secml.ml.classifiers.reject import CClassifierReject
from secml.core.exceptions import NotFittedError
from secml.array import CArray
from secml.optim.function import CFunction


class CModelCleverhans(Model):
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
        self._callable_fn = _CClassifierToTF(
            self._fun, self._out_dims)

        super(CModelCleverhans, self).__init__(nb_classes=clf.n_classes)

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
                cgrad = grad_f_x(
                    x_carray[0, :], y=c)
                grads[0, :] += (
                        cgrad * CArray(grads_in_np)[0, c])

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
        pyfun = tf.py_func(
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
    rnd_name = 'PyFuncGrad-' + '%0x' % getrandbits(30 * 4)

    # Register Tensorflow Gradient
    tf.RegisterGradient(rnd_name)(grad_func)

    # Get current graph
    g = tf.compat.v1.get_default_graph()

    # Add gradient override map
    with g.gradient_override_map(
            {"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=pyfun_name)
