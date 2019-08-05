"""
.. module:: ClassifierUtils
   :synopsis: Collection of mixed utilities for Classifiers

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.core.type_utils import is_int
from secml.array import CArray


def check_binary_labels(labels):
    """Check if input labels are binary {0, +1}.

    Parameters
    ----------
    labels : CArray or int
        Binary labels to be converted.
        As of PRALib convention, binary labels are {0, +1}.

    Raises
    ------
    ValueError
        If input labels are not binary.


    """
    if (is_int(labels) and not (labels == 0 or labels == 1)) or \
            (isinstance(labels, CArray) and
             (labels != 0).logical_and(labels != 1).any()):
        raise ValueError("input labels should be binary in {0, +1} interval.")


def convert_binary_labels(labels):
    """Convert input binary labels to {-1, +1}.

    Parameters
    ----------
    labels : CArray or int
        Binary labels in {0, +1} to be converted to {-1, +1}.

    Returns
    -------
    converted_labels : CArray or int
        Binary labels converted to {-1, +1}.

    Examples
    --------
    >>> from secml.ml.classifiers.clf_utils import convert_binary_labels
    >>> from secml.array import CArray

    >>> print(convert_binary_labels(0))
    -1

    >>> print(convert_binary_labels(CArray([0,1,1,1,0,0])))
    CArray([-1  1  1  1 -1 -1])

    """
    check_binary_labels(labels)
    return 2 * labels - 1
