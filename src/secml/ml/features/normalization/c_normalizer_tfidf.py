"""
.. module:: CNormalizerTFIDF
   :synopsis: Applies tf-idf normalization on a count matrix.

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""

from sklearn.feature_extraction.text import TfidfTransformer

from secml.array import CArray
from secml.ml.features.normalization import CNormalizer, CNormalizerUnitNorm


class CNormalizerTFIDF(CNormalizer):
    """
    Transform a count matrix to a normalized tf or tf-idf representation.

    Tf means term-frequency while tf-idf means term-frequency times inverse
    document-frequency. This is a common term weighting scheme in information
    retrieval, that has also found good use in document classification.

    The goal of using tf-idf instead of the raw frequencies of occurrence of a
    token in a given document is to scale down the impact of tokens that occur
    very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus.

    The formula that is used to compute the tf-idf for a term t of a document
    d in a document set is tf-idf(t, d) = tf(t, d) * idf(t), and the idf is
    computed as idf(t) = log [ (1 + n) / (1 + df(d, t)) ] + 1, where n
    is the total number of documents in the document set and df(t) is the
    document frequency of t; the document frequency is the number of documents
    in the document set that contains the term t. The effect of adding “1” to
    the idf in the equation above is that terms with zero idf, i.e., terms
    that occur in all documents in a training set, will not be entirely
    ignored. (Note that the idf formula above differs from the standard
    textbook notation that defines the idf as
    idf(t) = log [ n / (df(t) + 1) ]).

    Parameters
    ----------
    norm : ‘l1’, ‘l2’, ’max’ or None, optional (default=’l2’)
        Each output row will have unit norm, either: * ‘l2’: Sum of squares of
        vector elements is 1. The cosine similarity between two vectors is
        their dot product when l2 norm has been applied. * ‘l1’: Sum of
        absolute values of vector elements is 1. * ’max’ : maximum of absolute
        values of vector elements.

    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'tf-idf'

    Notes
    -----
    Differently from numpy, we manage flat vectors as 2-Dimensional of
    shape (1, array.size). This means that normalizing a flat vector is
    equivalent to transform array.atleast_2d(). To obtain a numpy-style
    normalization of flat vectors, transpose the array first.

    """
    __class_type = 'tf-idf'

    def __init__(self, norm='l2', preprocess=None):
        # init attributes
        self._norm = None
        self._cached_x_tfidf = None  # cached x after tfidf for gradient comp.
        self._unitnorm = CNormalizerUnitNorm()
        self._sklearn_tfidf = TfidfTransformer(
            norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False)

        super(CNormalizerTFIDF, self).__init__(preprocess=preprocess)
        # set norm
        self.norm = norm

    def _clear_cache(self):
        """Clears cached values within this class instance."""
        self._cached_x_tfidf = None
        super(CNormalizerTFIDF, self)._clear_cache()

    @property
    def _grad_requires_forward(self):
        return True

    @property
    def norm(self):
        """Type of norm used to normalize the tf-idf."""
        return self._norm

    @norm.setter
    def norm(self, value):
        """Set norm."""
        if value is not None:
            self._unitnorm.norm = value
        self._norm = value

    def _check_is_fitted(self):
        """Check if the preprocessor has been trained (fitted).

        Raises
        ------
        NotFittedError
            If the preprocessor is not fitted.

        """
        if not hasattr(self._sklearn_tfidf, 'idf_'):
            raise ValueError("The normalizer has not been trained.")

    def _fit(self, x, y=None):
        """Learn the normalizer.

        Parameters
        ----------
        x : CArray
            Array to be used as training set. Each row must correspond to
            one single pattern and each column is a different feature.
        y : CArray or None, optional
            Flat array with the label of each pattern.
            Can be None if not required by the pre-processing algorithm.

        Returns
        -------
        CNormalizerTFIDF
            Instance of the trained normalizer.
        """
        # this sets idf_ inside the sklearn normalizer
        x = x.atleast_2d()
        self._sklearn_tfidf.fit(x.get_data(), None)
        return self

    def _forward(self, x):
        """
        Apply the TF-IDF transform.

        Parameters
        ----------
        x : CArray
            Array with features to be transformed.

        Returns
        -------
        Array with normalized features.
        Shape of returned array is the same of the original array.

        """
        # transform data
        x = CArray(self._sklearn_tfidf.transform(x.get_data()))
        if self.norm is not None:  # apply unitnorm if set
            # store x after the tf-idf transformation (needed for grad. comp.)
            self._cached_x_tfidf = x.deepcopy()
            x = self._unitnorm.transform(x)
        return x

    def _backward(self, w=None):
        """
        Compute the gradient w.r.t. the input cached during the forward pass.

        Parameters
        ----------
        w : CArray or None, optional
            If CArray, will be left-multiplied to the gradient
            of the preprocessor.

        Returns
        -------
        gradient : CArray
            Gradient of the normalizer wrt input data.
            - if `w` is passed as input and is two-dimensional it will have
            shape (w.shape[0], x.shape[1]),
            - if `w` is a flat array it will be:
                an array of shape (x.shape[1], x.shape[1]) if the parameter
                norm is not None or a flat array of shape (x.shape[1],)
                if the parameter norm is equal to None;
        """
        grad = CArray(self._sklearn_tfidf.idf_)  # flat vector

        if self.norm is None:
            return w * grad if w is not None else grad
        else:
            self._unitnorm.forward(self._cached_x_tfidf, caching=True)
            # compute the gradient using the chain rule
            grad_unitnorm = self._unitnorm.backward(w)
            return grad_unitnorm * grad
