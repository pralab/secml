"""
.. module:: DensityEstimation
   :synopsis: Kernel density estimatio

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from sklearn.neighbors import KernelDensity

from secml.array import CArray
from secml.core import CCreator


class CDensityEstimation(CCreator):
    """Kernel Density Estimation

    Parameters
    ----------
    bandwidth : float, optional
        The bandwidth of the kernel. Default 1.
    algorithm : str, optional
        The tree algorithm to use. 
        Valid options are ['kd_tree'|'ball_tree'|'auto']. Default is 'auto'.
    kernel : str, optional
        The kernel to use. Valid kernels are
        ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine'].
        Default is 'gaussian'.
    metric : str, optional
        The distance metric to use. Note that not all metrics are valid
        with all algorithms. Refer to the documentation of BallTree and
        KDTree for a description of available algorithms. Note that the
        normalization of the density output is correct only for the Euclidean
        distance metric. Default is 'euclidean'.
    atol : float, optional
        The desired absolute tolerance of the result.
        A larger tolerance will generally lead to faster execution.
        Default is 0.
    rtol : float, optional
        The desired relative tolerance of the result.
        A larger tolerance will generally lead to faster execution.
        Default is 1E-8.
    breadth_first : bool, optional
        If true (default), use a breadth-first approach to the problem.
        Otherwise use a depth-first approach.
    leaf_size : int, optional
        Specify the leaf size of the underlying tree.
        See BallTree or KDTree for details. Default is 40.
    metric_params : dict, optional
        Additional parameters to be passed to the tree for use
        with the metric. For more information, see the documentation
        of BallTree or KDTree.

    """
    def __init__(self, bandwidth=1.0, algorithm='auto', kernel='gaussian',
                 metric='euclidean', atol=0, rtol=1e-8, breadth_first=True,
                 leaf_size=40, metric_params=None):

        self.bandwidth = bandwidth
        self.algorithm = algorithm
        self.kernel = kernel
        self.metric = metric
        self.atol = atol
        self.rtol = rtol
        self.breadth_first = breadth_first
        self.leaf_size = leaf_size
        self.metric_params = metric_params

    def estimate_density(self, x, n_points=1000):
        """Estimate density of input array.

        Returns
        -------
        x : CArray
            Arrays with coordinates used to estimate density.
        df : CArray
            Density function values.

        """
        kde = KernelDensity(
            bandwidth=self.bandwidth,
            algorithm=self.algorithm,
            kernel=self.kernel,
            metric=self.metric,
            atol=self.atol,
            rtol=self.rtol,
            breadth_first=self.breadth_first,
            leaf_size=self.leaf_size,
            metric_params=self.metric_params).fit(x.atleast_2d().get_data())

        x = CArray.linspace(x.min() * 1.01, x.max() * 1.01, n_points)
        x = x.atleast_2d().T

        df = CArray(kde.score_samples(x.get_data()))
        df = df.exp()

        return x, df
