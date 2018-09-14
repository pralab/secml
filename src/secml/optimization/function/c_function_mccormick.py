"""
.. module:: McCormickFunction
   :synopsis: McCormick function

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from c_function import CFunction
from secml.array import CArray


class CFunctionMcCormick(CFunction):
    """The McCormick function.

    2-Dimensional function.

    Global minimum f(x) = -1.9133 @ x = (-0.54719, -1.54719).

    Given by:
    .. math::

        f(x) = sin(x_0 + x_1) + (x_0 - x_1)^2 - 1.5 * x_0 + 2.5 * x_1 + 1


    """
    class_type = 'mc_cormick'

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

        """
        x = x.atleast_2d()
        if x.shape[1] != 2:
            raise ValueError(
                "McCormick function available for 2 dimensions only")

        # Split into 3 parts
        f1 = CArray(x[0] + x[1]).sin()
        f2 = (x[0] - x[1]) ** 2
        f3 = -1.5 * x[0] + 2.5 * x[1] + 1

        return f1 + f2 + f3

    def _grad(self, x):
        """McCormick function gradient wrt. point x."""
        x = x.atleast_2d()
        if x.shape[1] != 2:
            raise ValueError("Gradient of McCormick function "
                             "only available for 2 dimensions")
        # Computing gradient of each dimension
        grad1_1 = CArray(x[0] + x[1]).cos()
        grad1_2 = 2 * (x[0] - x[1])
        grad1_3 = -1.5
        grad2_1 = CArray(x[0] + x[1]).cos()
        grad2_2 = -2 * (x[0] - x[1])
        grad2_3 = 2.5

        grad1 = CArray(grad1_1 + grad1_2 + grad1_3)
        grad2 = CArray(grad2_1 + grad2_2 + grad2_3)

        return CArray.concatenate(grad1, grad2, axis=1)
