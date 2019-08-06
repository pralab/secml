"""
.. module:: CLossHinge
   :synopsis: Hinge Loss Functions

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.ml.classifiers.loss import CLossClassification
from secml.ml.classifiers.loss.c_loss import _check_binary_score
from secml.ml.classifiers.clf_utils import convert_binary_labels
from secml.array import CArray


class CLossHinge(CLossClassification):
    """Hinge Loss Function.

    The function computes the average distance between the model and
     the data using hinge loss, a one-sided metric that considers only
     prediction errors.

    Hinge loss is used in maximal margin classifiers such as
     support vector machines.

    After converting the labels to {-1, +1},
     then the hinge loss is defined as:

    .. math::

        L_\\text{Hinge}(y, s) = \\max \\left\\{ 1 - sy, 0 \\right\\}

    Attributes
    ----------
    class_type : 'hinge'
    suitable_for : 'classification'

    """
    __class_type = 'hinge'

    def loss(self, y_true, score, pos_label=1):
        """Computes the value of the hinge loss function.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            2-D array of shape (n_samples, n_classes) or 1-D flat array
            of shape (n_samples,). If 1-D array, the probabilities
            provided are assumed to be that of the positive class.
        pos_label : {0, 1}, optional
            The class wrt compute the loss function. Default 1.
            If `score` is a 1-D flat array, this parameter is ignored.

        Returns
        -------
        CArray
            Loss function. Vector-like array.

        """
        if pos_label not in (0, 1):
            raise ValueError("only {0, 1} are accepted for `pos_label`")

        y_true = convert_binary_labels(y_true).ravel()  # Convert to {-1, 1}
        score = _check_binary_score(score, pos_label)

        # max(0, 1 - y*s)
        h = 1.0 - y_true * score
        h[h < 0] = 0.0

        return h

    def dloss(self, y_true, score, pos_label=1):
        """Computes the derivative of the hinge loss function with respect to `score`.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            2-D array of shape (n_samples, n_classes) or 1-D flat array
            of shape (n_samples,). If 1-D array, the probabilities
            provided are assumed to be that of the positive class.
        pos_label : {0, 1}, optional
            The class wrt compute the loss function derivative. Default 1.
            If `score` is a 1-D flat array, this parameter is ignored.

        Returns
        -------
        CArray
            Derivative of the loss function. Vector-like array.

        """
        if pos_label not in (0, 1):
            raise ValueError("only {0, 1} are accepted for `pos_label`")

        y_true = convert_binary_labels(y_true).ravel()  # Convert to {-1, 1}
        score = _check_binary_score(score, pos_label)

        # 0 if (1 - y*s) < 0 else -y_true
        d = -y_true.astype(float)  # labels are generally int

        h = 1.0 - y_true * score
        d[h < 0] = 0.0

        return d


class CLossHingeSquared(CLossClassification):
    """Squared Hinge Loss Function.

    The function computes the average distance between the model and
    the data using hinge loss, a one-sided metric that considers only
    prediction errors.

    After converting the labels to {-1, +1}, then the hinge loss is defined as:

    .. math::

        L^2_\\text{Hinge} (y, s) =
                    {\\left( \\max \\left\\{ 1 - sy, 0 \\right\\} \\right)}^2

    Attributes
    ----------
    class_type : 'hinge-squared'
    suitable_for : 'classification'

    """
    __class_type = 'hinge-squared'

    def loss(self, y_true, score, pos_label=1):
        """Computes the value of the squared hinge loss function.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            2-D array of shape (n_samples, n_classes) or 1-D flat array
            of shape (n_samples,). If 1-D array, the probabilities
            provided are assumed to be that of the positive class.
        pos_label : {0, 1}, optional
            The class wrt compute the loss function. Default 1.
            If `score` is a 1-D flat array, this parameter is ignored.

        Returns
        -------
        CArray
            Loss function. Vector-like array.

        """
        if pos_label not in (0, 1):
            raise ValueError("only {0, 1} are accepted for `pos_label`")

        y_true = convert_binary_labels(y_true).ravel()  # Convert to {-1, 1}
        score = _check_binary_score(score, pos_label)

        # (max(0, 1 - y*s))^2
        h = 1.0 - y_true * score
        h[h < 0] = 0.0

        return h ** 2

    def dloss(self, y_true, score, pos_label=1):
        """Computes the derivative of the squared hinge loss function with respect to `score`.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            2-D array of shape (n_samples, n_classes) or 1-D flat array
            of shape (n_samples,). If 1-D array, the probabilities
            provided are assumed to be that of the positive class.
        pos_label : {0, 1}, optional
            The class wrt compute the loss function derivative. Default 1.
            If `score` is a 1-D flat array, this parameter is ignored.

        Returns
        -------
        CArray
            Derivative of the loss function. Vector-like array.

        """
        if pos_label not in (0, 1):
            raise ValueError("only {0, 1} are accepted for `pos_label`")

        y_true = convert_binary_labels(y_true).ravel()  # Convert to {-1, 1}
        score = _check_binary_score(score, pos_label)

        # 0 if (1 - y * s) < 0 else -2 * y * (1 - y * s)
        h = 1.0 - y_true * score
        d = -2.0 * y_true * h
        d[h < 0] = 0.0

        return d
