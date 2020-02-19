"""
.. module:: CNormalizerTFIDF
   :synopsis: Applies tf-idf normalization on a count matrix.

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

from secml.array import CArray
from secml.ml.features.normalization import CNormalizer


class CNormalizerTFIDF(CNormalizer):
    """
    Transform a count matrix to a normalized tf or tf-idf representation

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
    norm : ‘l1’, ‘l2’ or None, optional (default=’l2’)
        Each output row will have unit norm, either: * ‘l2’: Sum of squares of
        vector elements is 1. The cosine similarity between two vectors is
        their dot product when l2 norm has been applied. * ‘l1’: Sum of
        absolute values of vector elements is 1.

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
    normalization of flat vectors, transpose array first.
    """

    __class_type = 'tf-idf'

    def __init__(self, norm='l2', preprocess=None):

        self._norm = norm
        self._tfidf_norm = None
        super(CNormalizerTFIDF, self).__init__(preprocess=preprocess)

    @property
    def norm(self):
        """Type of norm used to normalize the tf-idf."""
        return self._norm

    def _check_is_fitted(self):
        """Checks if the preprocessor is trained (fitted).

        Raises
        ------
        NotFittedError
            If the preprocessor is not fitted.

        """
        if self._tfidf_norm is None:
            raise ValueError("The normalizer has not been trained.")

    def _document_frequency(self, X):
        """Counts the number of non-zero values for each feature in sparse X.

        Parameters
        ----------
        X : CArray
            Matrix containing the term frequencies.

         Returns
        -------
        Array with the document frequencies.
        """

        if X.issparse:
            df = CArray(np.bincount(np.array(X.nnz_indices[1]),
                                    minlength=X.shape[1]))
        else:
            bin_x = X.deepcopy()
            bin_x[bin_x > 0] = 1
            df = CArray(bin_x.sum(axis=0).ravel())

        return df

    def _get_norm(self, x, norm='l2'):
        """Compute the required norm on each row.

        Parameters
        ----------
        x : CArray
            Array of which we would like to compute the norm.
        norm: string 'l2' or 'l1' (l2 default)
            norm that we would like to compute.

        Returns
        -------
        Array with the computed norm.
        """

        if norm == 'l2':
            ord = 2
        elif norm == 'l1':
            ord = 1
        else:
            raise ValueError("unknown norm")

        norm = CArray(np.linalg.norm(x.tondarray(), axis=1, keepdims=True,
                                     ord=ord))

        # to avoid nan values
        norm[norm == 0] = 1

        return norm

    def _forward(self, x):
        """
        Applies the TF-IDF transform.

        Parameters
        ----------
        x : CArray
            Array with features to be scaled.

        Returns
        -------
        Array with normalized features.
        Shape of returned array is the same of the original array.

        """
        if x.atleast_2d().shape[1] != self._idf.size:
            raise ValueError("array to normalize must have {:} "
                             "features (columns).".format(self._idf.size))

        x = x.atleast_2d()

        tf_idf = x * self._idf
        self._unnorm_tf_idf = tf_idf

        if self.norm is not None:
            self._tf_idf_norm = self._get_norm(tf_idf, norm=self.norm)
            tf_idf = tf_idf / self._tf_idf_norm

        return tf_idf

    def _fit(self, x, y=None):
        """Learns the normalizer.

        Parameters
        ----------
        x : CArray
            Array to be used as training set. Each row must correspond to
            one single patterns, so each column is a different feature.
        y : CArray or None, optional
            Flat array with the label of each pattern.
            Can be None if not required by the preprocessing algorithm.

        Returns
        -------
        CNormalizerTFIDF
            Instance of the trained normalizer.
        """
        x = x.atleast_2d()
        self._tfidf_norm = TfidfTransformer(norm=self.norm,
                                            smooth_idf=True)
        self._tfidf_norm.fit(x.tondarray(), None)

        self._n = x.shape[0]
        self._df = self._document_frequency(x)
        self._idf = CArray(self._tfidf_norm.idf_)

        return self

    def _inverse_transform(self, x):
        """Undo the normalization of input data.

        Parameters
        ----------
        x : CArray
            Array to be reverted. Must have been normalized by the same
            calling transform.

        Returns
        -------
        original_array : CArray
            Array with features scaled back to original values.

        """
        x = x.deepcopy()
        if x.atleast_2d().shape[1] != self._idf.size:
            raise ValueError("array to revert must have {:} "
                             "features (columns).".format(self._idf.size))

        if self.norm is not None:
            x *= self._tf_idf_norm

        # avoids division by zero
        x[:, self._idf != 0] /= self._idf[self._idf != 0]

        x = x.ravel() if x.ndim <= 1 else x

        return x

    def _backward(self, w=None):
        """Computes the gradient wrt the cached inputs during the forward pass.

        Parameters
        ----------
        w : CArray or None, optional
            If CArray, will be left-multiplied to the gradient
            of the preprocessor.

        Returns
        -------
        gradient : CArray
            Gradient of the normalizer wrt input data.
            - a flat array of shape (x.shape[1], ) if `w` is None;
            - if `w` is passed as input, will have (w.shape[0], x.shape[1]),
              or (x.shape[1], ) if `w` is a flat array.

        """
        grad = self._idf

        if self.norm is None:
            return w * grad if w is not None else grad

        else:
            # todo: we can speed up this avoiding some computation if we
            # usually call this passing w

            # compute the gradient of norm(tfidf) w.r.t. x
            if self.norm == 'l2':
                grad_tfidf_x = self._cached_x / self._tf_idf_norm

            elif self.norm == 'l1':
                sign = self._unnorm_tf_idf.sign()
                sign[sign == 0] = 1
                grad_tfidf_x = sign

            else:
                raise ValueError("Norm type unknown")

            # the first term is zero for the element of the gradient not in
            # the diagonal
            grad = CArray.diag(
                grad) * self._tf_idf_norm - self._unnorm_tf_idf.T.dot(
                grad_tfidf_x)

            grad /= (self._tf_idf_norm ** 2)

            return w.dot(grad) if w is not None else grad
