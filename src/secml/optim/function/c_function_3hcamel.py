"""
.. module:: McCormickFunction
   :synopsis: McCormick function

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.optim.function import CFunction
from secml.array import CArray


class CFunctionThreeHumpCamel(CFunction):
    """The Three-Hump Camel function.

    2-Dimensional function.

    Global minimum f(x) = 0 @ x = (0, 0).

    Given by:
    .. math::

        f(x) = 2 * x_0 ** 2 - 1.05 * x_0 ** 4 +
         x_0 ** 6 / 6 + x_0 * x_1 + x_1 ^ 2

    Attributes
    ----------
    class_type : '3h-camel'

    """
    __class_type = '3h-camel'

    def __init__(self):

        # Passing data to CFunction
        super(CFunctionThreeHumpCamel, self).__init__(
            fun=self._fun, n_dim=2, gradient=self._grad)

    def _fun(self, x):
        """Apply Three-Hump Camel function to point x.

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
                "Three-Hump Camel function available for 2 dimensions only")

        # Split into 2 parts
        f1 = 2 * x[0].item() ** 2 - 1.05 * x[0].item() ** 4
        f2 = x[0].item() ** 6 / 6 + \
             x[0].item() * x[1].item() + x[1].item() ** 2

        return f1 + f2

    def _grad(self, x):
        """Three-Hump Camel function gradient wrt. point x."""
        x = x.atleast_2d()
        if x.shape[1] != 2:
            raise ValueError("Gradient of Three-Hump Camel function "
                             "only available for 2 dimensions")
        # Computing gradient of each dimension
        grad1_1 = 4 * x[0] - 4.2 * x[0] ** 3
        grad1_2 = x[0] ** 5 + x[1]
        grad2_1 = 0
        grad2_2 = x[0] + 2 * x[1]

        grad1 = grad1_1 + grad1_2
        grad2 = grad2_1 + grad2_2

        return CArray.concatenate(grad1, grad2, axis=1).ravel()

    @staticmethod
    def global_min():
        """Value of the global minimum of the function.

        Global minimum f(x) = 0 @ x = (0, 0).

        Returns
        -------
        float
            Value of the global minimum of the function.

        """
        return 0.

    @staticmethod
    def global_min_x():
        """Global minimum point of the function.

        Global minimum f(x) = 0 @ x = (0, 0).

        Returns
        -------
        CArray
            The global minimum point of the function.

        """
        return CArray([0., 0.])
