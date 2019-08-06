"""
.. module:: ListUtils
   :synopsis: Collection of mixed utilities for lists

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""


def find_duplicates(l):
    """Find and returns a python set with input list duplicates.

    Parameters
    ----------
    l : list
        List to examine.

    Returns
    -------
    duplicates : set
        Python set with input list duplicates.

    Examples
    --------
    >>> from secml.utils.list_utils import find_duplicates
    >>> l = ['1', '1', 2, '3', 2]
    >>> print(find_duplicates(l))
    set(['1', 2])

    References
    ----------
    http://stackoverflow.com/questions/9835762/find-and-list-duplicates-in-python-list

    """
    seen = set()
    seen2times = set()
    # Preallocating functions to add items rapidly
    seen_add = seen.add
    seen2times_add = seen2times.add
    for item in l:
        if item in seen:
            seen2times_add(item)
        else:
            seen_add(item)
    return seen2times
