"""
.. module:: LinearFunction
   :synopsis: Manager for linear function

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from c_function import CFunction
from prlib.array import CArray


class CFunctionLinear(CFunction):
    """Implements linear functions of the form:
        b' x + c = 0

    """
    class_type = 'linear'

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
        """Apply quadratic function to point x."""
        if x.ndim == 2 and x.shape[0] > 1:
            score = CArray.zeros(x.shape[0])
            for i in xrange(x.shape[0]):
                score[i] = x[i, :].dot(self._b) + self._c
            return score
        return CArray(x.dot(self._b) + self._c)

    def _linear_grad(self, x=None):
        """Implements gradient of linear function wrt point x."""
        return CArray(self._b.ravel())
