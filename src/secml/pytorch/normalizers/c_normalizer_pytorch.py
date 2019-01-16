"""
.. module:: CNormalizerPyTorch
   :synopsis: Normalizer which returns the deepfeatures at a specified neural network layer.

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.array import CArray
from secml.ml.features.normalization import CNormalizer


class CNormalizerPyTorch(CNormalizer):
    """Normalized features are the pytorch_clf deepfeatures

    Parameters
    ----------
    pytorch_clf : CClassifierPyTorch
        PyTorch classifier to be used for extracting deepfeatures.
        This must be already trained.
    out_layer : str or None, optional
        Identifier of the layer at which the features must be retrieved.
        If None, the output of last layer will be returned.

    Attributes
    ----------
    class_type : 'pytorch'

    """
    __class_type = 'pytorch'

    def __init__(self, pytorch_clf, out_layer=None):

        self._pytorch_clf = pytorch_clf
        self.out_layer = out_layer

    # DO NOT clear the inner pytorch_clf as must be passed already trained

    def __is_clear(self):
        """Returns True if object is clear."""
        return self.pytorch_clf.is_clear()

    @property
    def pytorch_clf(self):
        """The PyTorch Classifier."""
        return self._pytorch_clf

    def fit(self, x):
        """Fit normalization algorithm using data.

        This fit function is just a placeholder and simply returns
        the normalizer itself.

        Parameters
        ----------
        x : CArray
            Array to be used for training normalization algorithm.
            Shape of input array depends on the algorithm itself.

        Returns
        -------
        CNormalizer
            Instance of the normalization algorithm trained using
            input data.

        """
        return self

    def normalize(self, x):
        """Apply the normalization algorithm on data.

        This extracts the deepfeatures at the specified layer
        of the PyTorch neural network.

        Parameters
        ----------
        x : CArray
            Array to be normalized using normalization algorithm.

        Returns
        -------
        CArray
            Deepfeatures at the specified PyTorch neural network layer.
            Shape depends on the neural network layer shape.

        """
        # Training first!
        if self.is_clear() is True:
            raise ValueError(
                "fit the normalizer and the PyTorch classifier first.")

        return self.pytorch_clf.get_layer_output(x, self.out_layer)

    def gradient(self, x, y=None, w=None):
        """Returns the normalizer gradient wrt data.

        Parameters
        ----------
        x : CArray
            Data array, 2-Dimensional or ravel.
        y : int or None, optional
            Index of the class wrt the gradient must be computed.
            This is not required if w is passed.
        w : CArray or None, optional
            If CArray, will be passed to backward in the net and must have
            a proper shape depending on the chosen output layer.
            This is required if `out_layer` is not None.

        Returns
        -------
        gradient : CArray
            Gradient of the normalizer wrt input data.

        """
        # Training first!
        if self.is_clear() is True:
            raise ValueError(
                "fit the normalizer and the PyTorch classifier first.")

        if not x.is_vector_like:
            raise ValueError('Gradient available only wrt a single point!')

        return self.pytorch_clf.gradient_f_x(
            x, y=y, w=w, layer=self.out_layer)
