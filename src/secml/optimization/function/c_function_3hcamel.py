"""
.. module:: McCormickFunction
   :synopsis: McCormick function

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from c_function import CFunction
from prlib.array import CArray


class CFunctionThreeHumpCamel(CFunction):
    """The Three-Hump Camel function.

    2-Dimensional function.

    Global minimum f(x) = 0 @ x = (0, 0).

    Given by:
    .. math::

        f(x) = 2 * x_0 ** 2 - 1.05 * x_0 ** 4 +
         x_0 ** 6 / 6 + x_0 * x_1 + x_1 ^ 2


    """
    class_type = '3h_camel'

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

        """
        x = x.atleast_2d()
        if x.shape[1] != 2:
            raise ValueError(
                "Three-Hump Camel function available for 2 dimensions only")

        # Split into 2 parts
        f1 = 2 * x[0] ** 2 - 1.05 * x[0] ** 4
        f2 = float(x[0] ** 6) / 6 + x[0] * x[1] + x[1] ** 2

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

        grad1 = CArray(grad1_1 + grad1_2)
        grad2 = CArray(grad2_1 + grad2_2)

        return CArray.concatenate(grad1, grad2, axis=1)
