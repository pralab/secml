from secml.testing import CUnitTest

import numpy as np
from scipy.stats import norm

from secml.figure import CFigure
from secml.array import CArray
from secml.ml.stats import CDensityEstimation


class TestCClass(CUnitTest):
    """Unittests for CDensityEstimation."""

    def test_plot_density(self):

        N = 200
        np.random.seed(1)
        X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                            np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]

        X_plot = CArray(np.linspace(-5, 10, 1000)[:, np.newaxis])

        true_dens = CArray(0.3 * norm(0, 1).pdf(X_plot[:, 0].tondarray())
                           + 0.7 * norm(5, 1).pdf(X_plot[:, 0].tondarray()))

        fig = CFigure(width=7)
        fig.sp._sp.fill(X_plot[:, 0].tondarray(), true_dens.tondarray(),
                        fc='black', alpha=0.2,
                        label='input distribution')

        for kernel in ['gaussian', 'tophat', 'epanechnikov']:
            kde = CDensityEstimation(kernel=kernel, bandwidth=0.5)
            x, y = kde.estimate_density(CArray(X), n_points=N)
            fig.sp.plot(x, y, '-',
                        label="kernel = '{0}'".format(kernel))

        fig.sp.text(6, 0.38, "N={0} points".format(N))

        fig.sp.legend(loc='upper left')
        fig.sp.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

        fig.sp.xlim(-4, 9)
        fig.sp.ylim(-0.02, 0.4)
        fig.show()


if __name__ == '__main__':
    CUnitTest.main()
