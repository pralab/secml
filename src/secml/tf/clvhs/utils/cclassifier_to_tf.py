"""
.. module:: CClassifierToTfOperation
   :synopsis: Functionalities to create a Tensorflow operation from the
              discriminant function of a given classifier.

              Nb: The function "convert_cclassifier_to_tf" is generic and it
              works with each CClassifier, however it is really slow.
              If your CClassifier is based on PyTorch is better using the
              "convert_pytorch_cclassifier_to_tf" function.

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from random import getrandbits
from six.moves import range

import tensorflow as tf
import numpy as np

from secml.array import CArray


__all__ = ['convert_cclassifier_to_tf']


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
    grad : Custom Gradient Function.

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
    g = tf.get_default_graph()

    # Add gradient override map
    with g.gradient_override_map(
            {"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=pyfun_name)


def convert_cclassifier_to_tf(model, out_dims=1):
    """
    Creates a Tensorflow operation whose result are the scores produced by
    the discriminant function of a CClassifier subclass.
    Accordingly, the gradient of the discriminant function will be the gradient
    of the created Tensorflow operation.

    Parameters
    ----------
    model : A CClassifier object.
    out_dims : The number of output dimensions (classes) of the model.

    Returns
    -------
    tf_model_fn
        A Tensorfow operation that maps an input (tf.Tensor) to the output
        of model discriminant function (tf.Tensor) and on which the
        tensorflow function "gradient" can be called to compute its gradient.

    """

    def _fprop_fn(x_np):
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
        scores = model.predict(CArray(x_np), return_decision_function=True)[
            1].atleast_2d().tondarray().astype(np.float32)

        return scores

    def _np_grad_fn(x_np, grads_in_np=None):
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
        if grads_in_np.sum(axis=None) == 1:
            y = grads_in_np.find(grads_in_np == 1)[0]
            if grads_in_np[y] == 1:
                grads = model.gradient_f_x(x_carray, y=y).atleast_2d()
        else:
            # otherwise we have to compute the gradient w.r.t all the classes
            for c in range(n_classes):
                cgrad = model.gradient_f_x(
                    x_carray[0, :], y=c)
                grads[0, :] += (
                        cgrad * CArray(grads_in_np)[0, c])

        return grads.tondarray().astype(np.float32)

    def _tf_gradient_fn(op, grads_in):
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
        pyfun = tf.py_func(_np_grad_fn, [op.inputs[0], grads_in],
                           Tout=[tf.float32])
        return pyfun

    def tf_model_fn(x_op):
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
        out = _py_func_with_gradient(_fprop_fn, [x_op],
                                     Tout=[tf.float32],
                                     stateful=True,
                                     grad_func=_tf_gradient_fn)[0]
        out.set_shape([None, out_dims])

        return out

    return tf_model_fn
