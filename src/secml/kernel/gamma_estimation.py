"""
.. module:: GammaEstimation
   :synopsis: Heuristic for gamma estimation

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from prlib.array import CArray
from prlib.classifiers import CClassifierKNN


# TODO: DECIDE WHERE TO PUT THIS FUNCTION
def gamma_estimation(dataset, factor=0.3):
    """Select gamma for RBF kernel to properly enclose dataset clusters.

    Parameters
    ----------
    dataset : CDataset
        Dataset from which gamma should be computed.
    factor : float
        Factor to use for balancing the gamma estimation.
        Smaller is the factor, higher will be gamma and viceversa.

    Returns
    -------
    gamma : float
        Computed gamma value.

    """
    if factor <= 0 or factor > 1:
        raise ValueError("factor must be inside (0, 1] range.")
    # Smaller is the number of neighboors, higher is the gamma
    num_cons_neighboors = int(dataset.num_samples * factor)
    if num_cons_neighboors < dataset.num_classes:
        raise ValueError("dataset must have at least {:} samples".format(
            int(float(dataset.num_classes) / factor)))
    knn = CClassifierKNN(n_neighbors=num_cons_neighboors)
    knn.train(dataset)
    median_distances = CArray.zeros(dataset.num_samples)
    # For each dataset point get the median distance with the k-neighboors
    for s in xrange(dataset.num_samples):
        dist = knn.kneighbors(dataset.X[s, :], num_cons_neighboors)[0]
        median_distances[s] = dist.median(axis=None)
    # Use the median of the median distances as sigma
    sigma = median_distances.median(axis=None)
    return 1.0 / (2 * sigma ** 2)
