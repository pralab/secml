"""
.. module:: CFunctionLinear
   :synopsis: Linear function

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.optim.function import CFunction
from secml.array import CArray


class CFunctionLinear(CFunction):
    """Implements linear functions of the form:
        b' x + c = 0

    Attributes
    ----------
    class_type : 'linear'

    """
    __class_type = 'linear'

    def __init__(self, b, c):

        if b.ndim != 2 or b.shape[1] != 1:
            raise ValueError('b is not a column vector!')

        self._b = b
        self._c = c

        # Passing data to CFunction
        super(CFunctionLinear, self).__init__(fun=self._linear_fun,
                                              n_dim=b.shape[0],
                                              gradient=self._linear_grad)

    def _linear_fun(self, x):
        """Apply linear function to point x.

        Parameters
        ----------
        x : CArray
            Data point.

        Returns
        -------
        scalar
            Result of the function applied to input point.

        """
        return x.dot(self._b) + self._c

    def _linear_grad(self):
        """Implements gradient of linear function wrt point x."""
        return CArray(self._b.ravel())
