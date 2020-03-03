"""
.. module:: PrototypesSelectorKMedians
   :synopsis: Selector of prototypes using k-medians strategy.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.data.selection import CPrototypesSelector
from secml.array import CArray
from secml.ml.kernels import CKernelEuclidean


class CPSKMedians(CPrototypesSelector):
    """Selection of Prototypes using K-Medians strategy.

    Runs a k-means clustering to obtain a set of clusters from
    the dataset. Then selects the prototypes as their set medians.

    References
    ----------
    Spillmann, Barbara, et al. "Transforming strings to vector
    spaces using prototype selection." Structural, Syntactic,
    and Statistical Pattern Recognition.
    Springer Berlin Heidelberg, 2006. 287-296.

    Attributes
    ----------
    class_type : 'k-medians'

    """
    __class_type = 'k-medians'

    def select(self, dataset, n_prototypes, random_state=None):
        """Selects the prototypes from input dataset.

        Parameters
        ----------
        dataset : CDataset
            Dataset from which prototypes should be selected
        n_prototypes : int
            Number of prototypes to be selected.
        random_state : int, RandomState or None, optional
            Determines random number generation for centroid initialization.
            Default None.

        Returns
        -------
        reduced_ds : CDataset
            Dataset with selected prototypes.

        """
        from sklearn.cluster import k_means
        km = k_means(dataset.X.tondarray(), n_clusters=n_prototypes,
                     random_state=random_state)
        km_labels = CArray(km[1])
        # Precomputing distances
        k_euclidean = - CKernelEuclidean().k(dataset.X)
        # List of selected prototypes (indices)
        sel_idx = []
        for i in range(n_prototypes):
            # Find the samples associated with each cluster
            cluster_indices = km_labels.find(km_labels == i)
            if len(cluster_indices) == 0:  # No sample in the cluster?!
                raise ValueError("No sample in the cluster {:}".format(i))
            elif len(cluster_indices) == 1:  # One sample in the cluster
                p = 0
            else:  # Compute the median prototype
                p = k_euclidean[cluster_indices, cluster_indices]
                # Compute the median prototype
                p = p.sum(axis=0, keepdims=False).argmin()

            sel_idx.append(cluster_indices[p])

        self.logger.debug("Selecting samples: {:}".format(sel_idx))

        self._sel_idx = CArray(sel_idx)

        # Returning the reduced training set
        return dataset[self._sel_idx, :]
