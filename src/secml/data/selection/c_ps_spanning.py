"""
.. module:: PrototypesSelectorSpanning
   :synopsis: Selector of prototypes using spanning strategy.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.data.selection import CPrototypesSelector
from secml.array import CArray
from secml.ml.kernels import CKernelEuclidean


class CPSSpanning(CPrototypesSelector):
    """Selection of Prototypes using spanning strategy.

    Selects the first prototype as the dataset median, and the
    remaining ones iteratively, by maximizing the distance to
    the set of previously-selected prototypes.

    References
    ----------
    Spillmann, Barbara, et al. "Transforming strings to vector
    spaces using prototype selection." Structural, Syntactic,
    and Statistical Pattern Recognition.
    Springer Berlin Heidelberg, 2006. 287-296.

    Attributes
    ----------
    class_type : 'spanning'

    """
    __class_type = 'spanning'

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
        # Precomputing distances
        k_euclidean = - CKernelEuclidean().k(dataset.X)
        # List of selected prototypes (indices)
        # First sample is the median
        sel_idx = [k_euclidean.sum(axis=0, keepdims=False).argmin()]
        set_indices = list(range(dataset.num_samples))
        for i in range(1, n_prototypes):
            set_indices = [e for e in set_indices if e not in sel_idx]
            p = k_euclidean[set_indices, sel_idx]
            # Compute the farthest prototype
            sel_idx.append(set_indices[p.min(axis=1, keepdims=False).argmax()])

        self.logger.debug("Selecting samples: {:}".format(sel_idx))

        self._sel_idx = CArray(sel_idx)

        # Returning the reduced training set
        return dataset[self._sel_idx, :]
