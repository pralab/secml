"""
.. module:: DataUilts
   :synopsis: Collection of mixed utilities for data processing

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn.preprocessing import label_binarize as sk_binarizer
import numpy as np

from secml.array import CArray

__all__ = ['label_binarize_onehot']


def label_binarize_onehot(y):
    """Return dataset labels in one-hot encoding.

    Parameters
    ----------
    y : CArray
        Array with the labels to encode. Only integer labels are supported.

    Returns
    -------
    binary_labels : CArray
        A (num_samples, num_classes) array with the labels one-hot encoded.

    Examples
    --------
    >>> a = CArray([1,0,2,1])
    >>> print(label_binarize_onehot(a))
    CArray([[0 1 0]
     [1 0 0]
     [0 0 1]
     [0 1 0]])

    """
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("only integer labels are supported")
    classes = CArray.arange(y.max() + 1)
    return CArray(sk_binarizer(
        y.tondarray(), classes=classes.tondarray()))
