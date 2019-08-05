"""
.. module:: CLossSquare
   :synopsis: Squared Loss Functions

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.ml.classifiers.loss import CLossRegression, CLossClassification
from secml.ml.classifiers.loss.c_loss import _check_binary_score
from secml.ml.classifiers.clf_utils import convert_binary_labels


class CLossSquare(CLossClassification):
    """Square Loss Function.

    The square loss is defined as:

    .. math::

       L_\\text{Square}(y, s) = \\left( 1 - sy \\right)

    Attributes
    ----------
    class_type : 'square'
    suitable_for : 'classification'

    """
    __class_type = 'square'

    def loss(self, y_true, score, pos_label=1):
        """Computes the value of the squared epsilon-insensitive loss function.

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

        return (1.0 - y_true * score) ** 2

    def dloss(self, y_true, score, pos_label=1):
        """Computes the derivative of the square loss function with respect to `score`.

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

        return -2.0 * y_true * (1.0 - y_true * score)


class CLossQuadratic(CLossRegression):
    """Quadratic Loss Function (Ordinary Least Squares).

    The quadratic loss is defined as:

    .. math::

        L_\\text{Quadratic} (y, s) = \\frac{1}{2} {\\left( s - y \\right)}^2

    Attributes
    ----------
    class_type : 'quadratic'
    suitable_for : 'regression'

    """
    __class_type = 'quadratic'

    def loss(self, y_true, score):
        """Computes the value of the quadratic loss function.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            Vector-like array of shape (n_samples,).

        Returns
        -------
        CArray
            Loss function. Vector-like array.

        """
        if score.is_vector_like is False:
            raise ValueError("only a vector-like `score` array is supported.")

        # Ensure we work with vector-like arrays
        y_true = y_true.ravel()
        score = score.ravel()

        return 0.5 * ((score - y_true) ** 2)

    def dloss(self, y_true, score):
        """Computes the derivative of the quadratic loss function with respect to `score`.

        Parameters
        ----------
        y_true : CArray
            Ground truth (correct), targets. Vector-like array.
        score : CArray
            Outputs (predicted), targets.
            Vector-like array of shape (n_samples,).

        Returns
        -------
        CArray
            Derivative of the loss function. Vector-like array.

        """
        if score.is_vector_like is False:
            raise ValueError("only a vector-like `score` array is supported.")

        # Ensure we work with vector-like arrays
        y_true = y_true.ravel()
        score = score.ravel()

        return score - y_true
