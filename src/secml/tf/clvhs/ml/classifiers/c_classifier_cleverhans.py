"""
.. module:: CClassifierCleverhans
   :synopsis: Class that, given a CClassifier, wrap it into a CleverHans Model.
             We use this class to be able to run the cleverhans attacks on
            CleverHans classifiers.

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from cleverhans.model import Model

from secml.tf.clvhs.utils import convert_cclassifier_to_tf

from secml.ml.classifiers.reject import CClassifierReject
from secml.core.exceptions import NotFittedError


class CClassifierCleverhans(Model):
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

    def __init__(self, clf, out_dims=None):

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
        self._callable_fn = convert_cclassifier_to_tf(clf, self._out_dims)

        super(CClassifierCleverhans, self).__init__(nb_classes=clf.n_classes)

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
        return {self._output_layer: self._callable_fn(x, **kwargs)}

    class NoSuchLayerError(ValueError):
        """Raised when a layer that does not exist is requested."""
        pass
