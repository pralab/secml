"""
.. module:: GaussianDistribution
   :synopsis: A dataset with an array of patterns and corresponding labels

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from scipy.stats import multivariate_normal
from secml.array import CArray
from secml.core import CCreator


class CDistributionGaussian(CCreator):
    """A multivariate normal random variable.

    Parameters
    ----------
    mean : scalar, optional
        Mean of the distribution (default zero)
    cov : array_like or scalar, optional
        Covariance matrix of the distribution (default one)

    """

    def __init__(self, mean=0, cov=1):

        self.mean = mean
        self.cov = cov

    def pdf(self, data):
        """Probability density function.

        Parameters
        ----------
        data : CArray
            Quantiles, with the last axis of x denoting the components.

        Returns
        -------
        pdf: CArray
            Probability density function computed at input data.

        """
        cov = self.cov
        if isinstance(cov, CArray):
            cov = cov.tondarray()
        return CArray(multivariate_normal.pdf(data.tondarray(),
                                              self.mean, cov))

    def logpdf(self, data):
        """Log of the probability density function.

        Parameters
        ----------
        data : CArray
            Quantiles, with the last axis of x denoting the components.

        Returns
        -------
        pdf: CArray
            Probability density function computed at input data.

        """
        cov = self.cov
        if isinstance(cov, CArray):
            cov = cov.tondarray()
        return CArray(multivariate_normal.logpdf(data.tondarray(),
                                                 self.mean, cov))
