"""
.. module:: CClassifierGradientMixin
   :synopsis: Common interface for the implementations of the
              classifier gradients Mixin classes

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta


class CClassifierGradientMixin(metaclass=ABCMeta):
    """Abstract Mixin class that defines basic methods
     for classifier gradients."""

    # train derivatives:

    def hessian_tr_params(self, x, y):
        """Hessian of the training objective w.r.t. the classifier parameters.

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the training objective is computed.
        y :  CArray
            Dataset labels.

        """
        raise NotImplementedError

    def grad_f_params(self, x, y):
        """Derivative of the decision function w.r.t. the classifier parameters.

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the training objective is computed
        y : int
            Index of the class wrt the gradient must be computed.

        """
        raise NotImplementedError

    # FIXME: remove grad_l_params and include it in grad_train_obj_params
    # this is not easy. (removing this function we should manage for example
    # the presence of C, which is peculiar of some classifier in the code
    # that now uses this function, which should be general.)
    def grad_loss_params(self, x, y, loss=None):
        """Derivative of a given loss w.r.t. the classifier parameters.

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the loss is computed
        y :  CArray
            Dataset labels
        loss: None (default) or CLoss
            If the loss is equal to None (default) the classifier loss is used
            to compute the derivative.

        """
        raise NotImplementedError

    def grad_tr_params(self, x, y):
        """
        Derivative of the classifier training objective function w.r.t. the
        classifier parameters.

        Parameters
        ----------
        x : CArray
            Features of the dataset on which the training objective is computed.
        y :  CArray
            Dataset labels.

        """
        raise NotImplementedError

    # TODO: this is going to be removed and replaced to a call to gradient(x,w)
    #  as soon as all classifiers will have _backward implemented properly
    def grad_f_x(self, x, y, **kwargs):
        """
        Derivative of the classifier decision function w.r.t. an input sample

        Parameters
        ----------
        x : CArray
            features of the dataset on which the decision function is computed
        y :  CArray
            The label of the class wrt the function should be calculated.
        kwargs
            Optional arguments for the gradient method.
            See specific classifier for a full description.

        Returns
        -------
        gradient : CArray
            Gradient of the classifier's output wrt input. Vector-like array.

        """
        self._check_is_fitted()

        x_in = x  # Original data

        # If preprocess is defined, transform data before computing the grad
        x = self._preprocess_data(x)

        try:  # Get the derivative of decision_function
            grad_f = self._grad_f_x(x, y, **kwargs)
        except NotImplementedError:
            raise NotImplementedError("{:} does not implement `grad_f_x`"
                                      "".format(self.__class__.__name__))

        # The derivative of decision_function should be a vector
        # as we are computing the gradient wrt a class `y`
        if not grad_f.is_vector_like:
            raise ValueError("`_gradient_f` must return a vector like array")

        grad_f = grad_f.ravel()

        # backpropagate the clf gradient to the preprocess (if defined)
        if self.preprocess is not None:
            # preprocess gradient will be accumulated in grad_f
            # and a vector-like array should be returned
            grad_p = self.preprocess.gradient(x_in, w=grad_f)
            if not grad_p.is_vector_like:
                raise ValueError(
                    "`preprocess.gradient` must return a vector like array")
            return grad_p.ravel()

        return grad_f  # No preprocess defined... return the clf grad
