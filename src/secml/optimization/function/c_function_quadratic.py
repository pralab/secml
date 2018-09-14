"""
.. module:: QuadraticFunction
   :synopsis: Manager for quadratic function

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
.. moduleauthor:: Paolo Russu <paolo.russu@diee.unica.it>

"""
from c_function import CFunction
from secml.array import CArray


class CFunctionQuadratic(CFunction):
    """Implements quadratic functions of the form:
        x' A x + b' x + c = 0

    """

    class_type = 'quadratic'

    def __init__(self, A, b, c):

        if len(A.shape) != 2:
            raise ValueError('A is not a 2D matrix!')
        elif A.shape[0] != A.shape[1]:
            raise ValueError('A is not a squared matrix!')

        # TODO: Add check: A should be symmetric as well

        if len(b.shape) != 2 or b.shape[1] != 1:
            raise ValueError('b is not a column vector!')

        if b.shape[0] != A.shape[0]:
            raise ValueError(
                'A and b have inconsistent dimensions!')

        self._A = A
        self._b = b
        self._c = c

        # Passing data to CFunction
        super(CFunctionQuadratic, self).__init__(fun=self._quadratic_fun,
                                                 n_dim=A.shape[0],
                                                 gradient=self._quadratic_grad)

    def _quadratic_fun(self, x):
        """Apply quadratic function to point x."""
        if x.ndim == 2 and x.shape[0] > 1:
            score = CArray.zeros(x.shape[0])
            for i in xrange(x.shape[0]):
                score[i] = x[i, :].dot(self._A).dot(
                    x[i, :].T) + x[i, :].dot(self._b) + self._c
            return score
        return CArray(x.dot(self._A).dot(x.T) + x.dot(self._b) + self._c)

    def _quadratic_grad(self, x):
        """Implements gradient of quadratic function wrt point x."""
        return CArray(2 * x.dot(self._A) + self._b.T).ravel()
