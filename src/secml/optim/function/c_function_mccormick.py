"""
.. module:: CFunctionMcCormick
   :synopsis: McCormick function

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.optim.function import CFunction
from secml.array import CArray


class CFunctionMcCormick(CFunction):
    """The McCormick function.

    2-Dimensional function.

    Global minimum f(x) = -1.9132 @ x = (-0.5472, -1.5472).
    This is on a compact domain (lb=[-1.5,-3], ub=[4,4])

    Given by:
    .. math::

        f(x) = sin(x_0 + x_1) + (x_0 - x_1)^2 - 1.5 * x_0 + 2.5 * x_1 + 1

    Attributes
    ----------
    class_type : mc-cormick'

    """
    __class_type = 'mc-cormick'

    def __init__(self):

        # Passing data to CFunction
        super(CFunctionMcCormick, self).__init__(
            fun=self._fun, n_dim=2, gradient=self._grad)

    def _fun(self, x):
        """Apply McCormick function to point x.

        Parameters
        ----------
        x : CArray
            Data point.

        Returns
        -------
        float
            Result of the function applied to input point.

        """
        x = x.atleast_2d()
        if x.shape[1] != 2:
            raise ValueError(
                "McCormick function available for 2 dimensions only")

        # Split into 3 parts
        f1 = (x[0] + x[1]).sin().item()
        f2 = (x[0].item() - x[1].item()) ** 2
        f3 = -1.5 * x[0].item() + 2.5 * x[1].item() + 1

        return f1 + f2 + f3

    def _grad(self, x):
        """McCormick function gradient wrt. point x."""
        x = x.atleast_2d()
        if x.shape[1] != 2:
            raise ValueError("Gradient of McCormick function "
                             "only available for 2 dimensions")
        # Computing gradient of each dimension
        grad1_1 = (x[0] + x[1]).cos()
        grad1_2 = 2 * (x[0] - x[1])
        grad1_3 = -1.5
        grad2_1 = (x[0] + x[1]).cos()
        grad2_2 = -2 * (x[0] - x[1])
        grad2_3 = 2.5

        grad1 = grad1_1 + grad1_2 + grad1_3
        grad2 = grad2_1 + grad2_2 + grad2_3

        return CArray.concatenate(grad1, grad2, axis=1).ravel()

    @staticmethod
    def global_min():
        """Value of the global minimum of the function.

        Global minimum f(x) = -1.9132 @ x = (-0.5472, -1.5472).

        Returns
        -------
        float
            Value of the global minimum of the function.

        """
        return -1.9132

    @staticmethod
    def global_min_x():
        """Global minimum point of the function.

        Global minimum f(x) = -1.9132 @ x = (-0.5472, -1.5472).

        Returns
        -------
        CArray
            The global minimum point of the function.

        """
        return CArray([-0.5472, -1.5472])

    @staticmethod
    def bounds():
        lb = CArray([-1.5, -3])
        ub = CArray([4, 4])
        return lb, ub
