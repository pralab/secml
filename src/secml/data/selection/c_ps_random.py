"""
.. module:: PrototypesSelectorRandom
   :synopsis: Selector of prototypes using spanning strategy.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.data.selection import CPrototypesSelector
from secml.array import CArray


class CPSRandom(CPrototypesSelector):
    """Selection of Prototypes using random strategy.

    Attributes
    ----------
    class_type : 'random'

    """
    __class_type = 'random'

    def select(self, dataset, n_prototypes, random_state=None):
        """Selects the prototypes from input dataset.

        Parameters
        ----------
        dataset : CDataset
            Dataset from which prototypes should be selected
        n_prototypes : int
            Number of prototypes to be selected.
        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, is the RandomState instance used by np.random.

        Returns
        -------
        reduced_ds : CDataset
            Dataset with selected prototypes.

        """
        sel_idx = CArray.randsample(CArray(list(range(dataset.num_samples))),
                                    shape=n_prototypes,
                                    random_state=random_state)

        self.logger.debug("Selecting samples: {:}".format(sel_idx.tolist()))

        self._sel_idx = sel_idx

        # Returning the reduced training set
        return dataset[self._sel_idx, :]
