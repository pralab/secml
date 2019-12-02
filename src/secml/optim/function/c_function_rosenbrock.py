"""
.. module:: CFunctionRosenbrock
   :synopsis: Rosenbrock function

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.optim.function import CFunction
from secml.array import CArray


class CFunctionRosenbrock(CFunction):
    """The Rosenbrock function.

    Non-convex function introduced by Howard H. Rosenbrock in 1960. [1]_
    Also known as Rosenbrock's valley or Rosenbrock's banana function.

    Global minimum f(x) = 0 @ x = (1, 1, ...., 1).

    Given by:
    .. math::

       f(x) = \\sum^{n-1}_{i=1} [100 * {(x_{i+1} - x_i^2)}^2 + (x_i - 1)^2]

    Attributes
    ----------
    class_type : 'rosenbrock'

    References
    ----------
    .. [1] Rosenbrock, HoHo. "An automatic method for finding
       the greatest or least value of a function." The Computer
       Journal 3.3 (1960): 175-184.

    """
    __class_type = 'rosenbrock'

    def __init__(self):

        # Passing data to CFunction
        super(CFunctionRosenbrock, self).__init__(fun=self._fun,
                                                  n_dim=None,
                                                  gradient=self._grad)

    def _fun(self, x):
        """Apply Rosenbrock function to point x.

        Parameters
        ----------
        x : CArray
            Data point.

        Returns
        -------
        scalar
            Result of the function applied to input point.

        """
        x = x.atleast_2d()
        if x.shape[1] < 2:
            raise ValueError(
                "Rosenbrock function available for at least 2 dimensions")

        f = 0  # Starting value
        for n in range(x.shape[1] - 1):
            f += 100 * (x[n+1].item() - x[n].item() ** 2) ** 2 + \
                 (x[n].item() - 1) ** 2

        return f

    def _grad(self, x):
        """Rosenbrock function gradient wrt. point x.

        Gradient available for 2-Dimensional points only.

        """
        x = x.atleast_2d()
        if x.shape[1] != 2:
            raise ValueError("Gradient of Rosenbrock function "
                             "only available for 2 dimensions")
        # Computing gradient of each dimension
        grad1 = -400 * (x[1] - x[0] ** 2) * x[0] + 2 * (x[0] - 1)
        grad2 = 200 * (x[1] - x[0] ** 2)

        return CArray.concatenate(grad1, grad2, axis=1).ravel()

    @staticmethod
    def global_min():
        """Value of the global minimum of the function.

        Global minimum f(x) = 0 @ x = (1, 1, ...., 1).

        Returns
        -------
        float
            Value of the global minimum of the function.

        """
        return 0.

    @staticmethod
    def global_min_x(ndim=2):
        """Global minimum point of the function.

        Global minimum f(x) = 0 @ x = (1, 1, ...., 1).

        Parameters
        ----------
        ndim : int, optional
            Number of dimensions to consider, higher or equal to 2. Default 2.

        Returns
        -------
        CArray
            The global minimum point of the function.

        """
        if ndim < 2:
            raise ValueError(
                "Rosenbrock function available for at least 2 dimensions")

        return CArray.ones((ndim, ), dtype=float)
