"""
.. module:: ClassifierUtils
   :synopsis: Collection of mixed utilities for Classifiers

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.array import CArray


def check_binary_labels(labels):
    """Check if input labels are binary {0, +1}.

    Parameters
    ----------
    labels : CArray
        Binary labels to be converted.
        As of PRALib convention, binary labels are {0, +1}.

    Raises
    ------
    ValueError
        If input labels are not binary.


    """
    if CArray(CArray(labels != 0).logical_and(labels != 1)).any():
        raise ValueError("input labels should be binary in 0/1 interval.")


def extend_binary_labels(labels):
    """Convert input binary labels to {-1, +1}.

    Parameters
    ----------
    labels : CArray
        Binary labels to be converted.
        As of PRALib convention, binary labels are {0, +1}.

    Returns
    -------
    converted_labels : CArray
        Binary labels converted to -1/+1.

    Examples
    --------
    >>> from secml.classifiers.clf_utils import extend_binary_labels
    >>> from secml.array import CArray

    >>> print extend_binary_labels(CArray([0,1,1,1,0,0]))
    CArray([-1  1  1  1 -1 -1])

    """
    check_binary_labels(labels)
    return 2 * labels - 1
