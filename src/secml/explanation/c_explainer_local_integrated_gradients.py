"""
.. module:: CExplainerLocalIntegratedGradients
   :synopsis: Integrated Gradients method for Local Explanation of predictions.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from __future__ import division
from secml.explanation import CExplainer
from secml.array import CArray
from secml.core.type_utils import is_int


class CExplainerLocalIntegratedGradients(CExplainer):
    """Integrated Gradients method for Local Explanation of predictions.

    This implements a method for local explanation of predictions
    via attribution of relevance to each feature.

    The algorithm takes a sample and computes the Riemman approximation
    of the integral along the linear interpolation with a reference point.

    - Sundararajan, Mukund, Ankur Taly, and Qiqi Yan.
     "Axiomatic Attribution for Deep Networks."
     arXiv preprint arXiv:1703.01365 (2017). https://arxiv.org/abs/1703.01365

    So we have for each dimension `i` of the input sample x:
    .. math::
        IG_i(x) = (x_i - x'_i) \times \sum^m_{k=1}
                   \frac{\partial F(x' + \frac{k}{m}\times(x-x'))}
                        {\partial x_i} \times \frac{1}{m}

    with `m` the number of steps in the Riemman approximation of the integral.

    Parameters
    ----------
    clf : CClassifier
        Instance of the classifier to explain.
    tr_ds : CDataset
        Training dataset of the classifier to explain.

    Attributes
    ----------
    class_type : 'integrated-gradients'

    """
    __class_type = 'integrated-gradients'

    def explain(self, x, reference=None, m=50, classes='all'):
        """Computes the explanation for input sample.

        Parameters
        ----------
        x : CArray
            Input sample.
        reference : CArray or None, optional
            The reference sample. Must have the same shape of input sample.
            If None, a all-zeros sample will be used.
        m : int, optional
            The number of steps for linear interpolation. Default 50.
        classes : CArray or int or str, optional
            CArray with the classes wrt the attributions should be computed.
            Can be a single class (int). If 'str' (default), all training
            classes will be considered.

        Returns
        -------
        attributions : CArray
            Attributions (weight of each feature) for input sample.
            Will be a 2D array with the attributions computed wrt each
            class in `classes` for each row.

        """
        if reference is None:
            # Use default reference values if reference is not specified
            reference = CArray.zeros(shape=x.shape, dtype=x.dtype)

        x = x.atleast_2d()

        if classes == 'all':  # Consider all training classes
            classes = self.clf.classes
        elif is_int(classes):
            classes = [classes]
        elif not isinstance(classes, CArray):
            raise TypeError("`classes` can be a CArray with the list of "
                            "classes, a single class id or 'all'")

        # Compute the linear interpolation from reference to input
        ret = self.linearly_interpolate(x, reference, m)

        attr = CArray.empty(shape=(0, x.shape[1]), dtype=x.dtype)
        for c in classes:  # Compute attributions for each class

            if c not in self.clf.classes:
                raise ValueError(
                    "class to explain {:} is invalid".format(c))

            # Compute the Riemman approximation of the integral
            riemman_approx = CArray.zeros(x.shape, dtype=x.dtype)
            for i in range(len(ret)):
                riemman_approx += self.clf.grad_f_x(ret[i], y=c)

            a = (x - reference) * (1 / m) * riemman_approx

            self.logger.debug(
                "Attributions for class {:}:\n{:}".format(c, a))

            # Checks prop 1: attr should adds up to the difference between
            # the score at the input and that at the reference
            self.check_attributions(x, reference, c, a)

            attr = attr.append(a, axis=0)

        return attr

    def check_attributions(self, x, reference, c, attributions):
        """Check proposition 1 on attributions.

        Proposition 1:
         Attributions should adds up to the difference between
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
        # Checks prop 1: attr should adds up to the difference between
        # the score at the input and that at the reference
        x_pred, x_score = self.clf.predict(
            x, return_decision_function=True)
        ref_pred, ref_score = self.clf.predict(
            reference, return_decision_function=True)
        prop_check = abs(x_score[c] - ref_score[c])
        prop_check = abs(prop_check - abs(attributions.sum())).item()
        if prop_check > 1e-1:
            self.logger.warning(
                "Attributions should adds up to the difference between the "
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
            reference = CArray.zeros(shape=x.shape, dtype=x.dtype)

        if x.shape != reference.shape:
            raise ValueError("reference must have shape {:}".format(x.shape))

        # Calculated stepwise difference from reference to the actual sample
        ret = []
        for s in range(1, m + 1):
            ret.append(reference + (x - reference) * (s * 1 / m))

        return ret
