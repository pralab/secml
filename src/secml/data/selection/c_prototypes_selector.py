"""
.. module:: PrototypesSelector
   :synopsis: Selector of prototypes to be used for Classification/Regression

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta, abstractmethod

from secml.core import CCreator


class CPrototypesSelector(CCreator, metaclass=ABCMeta):
    """Selection of Prototypes.

    Prototype selection methods help reducing the number of samples
    in a dataset by carefully selecting a subset of prototypes.

    [1]_ A good selection strategy should satisfy the following
    three conditions. First, if some prototypes are similar-that is,
    if they are close in the space of strings-their distances to a
    sample string should vary only little. Hence, in this case, some
    of the respective vector components are redundant. Consequently,
    a selection algorithm should avoid redundancies. Secondly, to
    include as much structural information as possible in the prototypes,
    they should be uniformly distributed over the whole set of patterns.
    Thirdly, since outliers are likely to introduce noise and distortions,
    a selection algorithm should disregard outliers.

    References
    ----------
    .. [1] Spillmann, Barbara, et al. "Transforming strings to vector spaces
       using prototype selection." Structural, Syntactic, and Statistical
       Pattern Recognition. Springer Berlin Heidelberg, 2006. 287-296.

    """
    __super__ = 'CPrototypesSelector'

    def __init__(self):

        self._sel_idx = None

    @property
    def sel_idx(self):
        """Returns an array with the indices of the selected prototypes."""
        return self._sel_idx

    @abstractmethod
    def select(self, dataset, n_prototypes):
        """Selects the prototypes from input dataset.

        Parameters
        ----------
        dataset : CDataset
            Dataset from which prototypes should be selected
        n_prototypes : int
            Number of prototypes to be selected.

        Returns
        -------
        reduced_ds : CDataset
            Dataset with selected prototypes.

        """
        raise NotImplementedError("Please implement a `select` method for "
                                  "class {:}".format(self.__class__.__name__))
