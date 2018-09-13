"""
.. module:: ClassifierUtils
   :synopsis: Collection of mixed utilities for Classifiers

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from prlib.array import CArray


def extend_binary_labels(labels):
    """Returns input binary labels to -1/+1.

    Parameters
    ----------
    labels : CArray
        Binary labels to be converted.
        As of PRALib convention, binary labels are in 0/+1 interval.

    Returns
    -------
    converted_labels : CArray
        Binary labels converted to -1/+1.

    Examples
    --------
    >>> from prlib.classifiers.clf_utils import extend_binary_labels
    >>> from prlib.array import CArray

    >>> print extend_binary_labels(CArray([0,1,1,1,0,0]))
    CArray([-1  1  1  1 -1 -1])

    """
    if CArray(CArray(labels != 0).logical_and(labels != 1)).any():
        raise ValueError("input labels should be binary in 0/1 interval.")

    return 2 * labels - 1
