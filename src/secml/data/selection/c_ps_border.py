"""
.. module:: PrototypesSelectorBorder
   :synopsis: Selector of prototypes using border strategy.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.data.selection import CPrototypesSelector
from secml.array import CArray
from secml.ml.kernels import CKernelEuclidean


class CPSBorder(CPrototypesSelector):
    """Selection of Prototypes using border strategy.

    Selects the prototypes from the borders of the dataset.

    References
    ----------
    Spillmann, Barbara, et al. "Transforming strings to vector
    spaces using prototype selection." Structural, Syntactic,
    and Statistical Pattern Recognition.
    Springer Berlin Heidelberg, 2006. 287-296.

    Attributes
    ----------
    class_type : 'border'

    """
    __class_type = 'border'

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
        sel_idx = []
        set_indices = list(range(dataset.num_samples))
        for i in range(n_prototypes):
            set_indices = [e for e in set_indices if e not in sel_idx]
            p = k_euclidean[set_indices, set_indices]
            # Compute the marginal prototype
            sel_idx.append(set_indices[p.sum(axis=0, keepdims=False).argmax()])

        self.logger.debug("Selecting samples: {:}".format(sel_idx))

        self._sel_idx = CArray(sel_idx)

        # Returning the reduced training set
        return dataset[self._sel_idx, :]
