"""
.. module:: CExplainerIntegratedGradients
   :synopsis: Integrated Gradients method for explanation of predictions.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray
from secml import _NoValue

from secml.explanation import CExplainerGradient


class CExplainerIntegratedGradients(CExplainerGradient):
    """Explanation of predictions via integrated gradients.

    This implements a method for local explanation of predictions
    via attribution of relevance to each feature.

    The algorithm takes a sample and computes the Riemman approximation
    of the integral along the linear interpolation with a reference point.

    - Sundararajan, Mukund, Ankur Taly, and Qiqi Yan.
      "Axiomatic Attribution for Deep Networks."
      Proceedings of the 34th International Conference on Machine Learning,
      Volume 70, JMLR. org, 2017, pp. 3319-3328.

    So we have for each dimension `i` of the input sample x:

    .. math::
       IG_i(x) = (x_i - x'_i) \\times \\sum^m_{k=1}
        \\frac{\\partial F(x' + \\frac{k}{m}\\times(x-x'))}
        {\\partial x_i} \\times \\frac{1}{m}

    with `m` the number of steps in the Riemman approximation of the integral.

    Parameters
    ----------
    clf : CClassifier
        Instance of the classifier to explain. Must be differentiable.

    Attributes
    ----------
    class_type : 'integrated-gradients'

    """
    __class_type = 'integrated-gradients'

    def explain(self, x, y, return_grad=_NoValue, reference=None, m=50):
        """Computes the explanation for input sample.

        Parameters
        ----------
        x : CArray
            Input sample.
        y : int
            Class wrt compute the classifier gradient.
        reference : CArray or None, optional
            The reference sample. Must have the same shape of input sample.
            If None, a all-zeros sample will be used.
        m : int, optional
            The number of steps for linear interpolation. Default 50.+

        Returns
        -------
        attributions : CArray
            Attributions (weight of each feature) for input sample.

        """
        if return_grad is not _NoValue:
            raise ValueError("`return_grad` is not supported by `{:}`".format(
                self.__class__.__name__))

        if reference is None:
            # Use default reference values if reference is not specified
            reference = CArray.zeros(
                shape=x.shape, dtype=x.dtype, sparse=x.issparse)

        x = x.atleast_2d()

        # Compute the linear interpolation from reference to input
        ret = self.linearly_interpolate(x, reference, m)

        # Compute the Riemann approximation of the integral
        riemman_approx = CArray.zeros(x.shape, sparse=x.issparse)
        for i in range(len(ret)):
            riemman_approx += self.clf.grad_f_x(ret[i], y=y)

        a = (x - reference) * (1 / m) * riemman_approx

        self.logger.debug(
            "Attributions for class {:}:\n{:}".format(y, a))

        # Checks prop 1: attr should add up to the difference between
        # the score at the input and that at the reference
        self.check_attributions(x, reference, y, a)

        return a

    def check_attributions(self, x, reference, c, attributions):
        """Check proposition 1 on attributions.

        Proposition 1:
         Attributions should add up to the difference between
         the score at the input and that at the reference point.

        Parameters
        ----------
        x : CArray
            Input sample.
        reference : CArray
            The reference sample. Must have the same shape of input sample.
        c : int
            Class wrt the attributions have been computed.
        attributions : CArray
            Attributions for sample `x` to check.

        """
        # Checks prop 1: attr should add up to the difference between
        # the score at the input and that at the reference
        x_pred, x_score = self.clf.predict(
            x, return_decision_function=True)
        ref_pred, ref_score = self.clf.predict(
            reference, return_decision_function=True)
        prop_check = abs(x_score[c] - ref_score[c])
        prop_check = abs(prop_check - abs(attributions.sum())).item()
        if prop_check > 1e-1:
            self.logger.warning(
                "Attributions should add up to the difference between the "
                "score at the input and that at the reference. Increase `m` "
                "or change the reference. Current value {:}.".format(prop_check))

    @staticmethod
    def linearly_interpolate(x, reference=None, m=50):
        """Computes the linear interpolation between the sample and the reference.

        Parameters
        ----------
        x : CArray
            Input sample.
        reference : CArray or None, optional
            The reference sample. Must have the same shape of input sample.
            If None, a all-zeros sample will be used.
        m : int, optional
            The number of steps for linear interpolation. Default 50.

        Returns
        -------
        list
            List of CArrays to integrate over.

        """
        if reference is None:
            # Use default reference values if reference is not specified
            reference = CArray.zeros(
                shape=x.shape, dtype=x.dtype, sparse=x.issparse)

        if x.shape != reference.shape:
            raise ValueError("reference must have shape {:}".format(x.shape))

        # Calculated stepwise difference from reference to the actual sample
        ret = []
        for s in range(1, m + 1):
            ret.append(reference + (x - reference) * (s * 1 / m))

        return ret
