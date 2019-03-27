"""
.. module:: CConstraintL1
   :synopsis: L1 Constraint

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from secml.array import CArray
from secml.optim.constraints import CConstraint


class CConstraintL1(CConstraint):
    """L1 Constraint.

    Attributes
    ----------
    class_type : 'l1'

    """
    __class_type = 'l1'

    def __init__(self, center=0, radius=1):

        super(CConstraintL1, self).__init__()

        self.center = center
        self.radius = radius

    @property
    def center(self):
        """Returns constraint L1 center."""
        return self._center

    @center.setter
    def center(self, value):
        """Sets constraint L1 center."""
        self._center = CArray(value)

    @property
    def radius(self):
        """Returns the semidiagonal of the constraint."""
        return self._radius

    @radius.setter
    def radius(self, value):
        """Sets the semidiagonal of the constraint."""
        self._radius = value

    def _constraint(self, x):
        """Returns the value of the constraint for the sample x.

        The constraint value y is given by:
         y = ||x - center||_1 - radius

        Parameters
        ----------
        x : CArray
            Flat 1-D array with the sample.

        Returns
        -------
        float
            Value of the constraint.

        """
        return float((x - self._center).norm(order=1) - self._radius)

    def _gradient(self, x):
        return (x - self._center).ravel().sign()

    def _projection(self, x):
        """ Compute the Euclidean projection on a L1-ball.

        Solves the optimisation problem (using the algorithm from [1]):
            min_w 0.5 * || w - x ||_2^2 , s.t. || w ||_1 <= s

        Parameters
        ----------
        x : CArray
            1-Dimensional array.

        Returns
        -------
        w : CArray
            Euclidean projection of v on the L1-ball of radius s.

        Notes
        -----
        Solves the problem by a reduction to the positive simplex case.

        """
        s = float(self._radius)
        v = (x - self._center).ravel()
        # compute the vector of absolute values
        u = abs(v)
        # check if v is already a solution
        if u.sum() <= s:
            # l1-norm is <= s
            return v + self._center

        # v is not already a solution: optimum lies on the boundary (norm == s)
        # project *u* on the simplex
        w = self._euclidean_proj_simplex(u, s=s)
        # compute the solution to the original problem on v
        w *= v.sign()
        return w + self._center

    def _euclidean_proj_simplex(self, v, s=1):
        """Compute the Euclidean projection on a positive simplex.

        Solves the optimisation problem (using the algorithm from [1]):

            min_w 0.5 * || w - v ||_2^2 ,
            s.t. \sum_i w_i = s, w_i >= 0

        Parameters
        ----------
        v : CArray
            1-Dimensional vector

        s : int, optional
            Radius of the simplex. Default 1.

        Returns
        -------
        w : CArray
           Euclidean projection of v on the simplex.

        Notes
        -----
        The complexity of this algorithm is in O(n log(n)) as it involves
        sorting v. Better alternatives exist for high-dimensional sparse
        vectors (cf. [1]). However, this implementation still easily
        scales to millions of dimensions.

        References
        ----------
        [1] Efficient Projections onto the l1-Ball for
            Learning in High Dimensions
            John Duchi, Shai Shalev-Shwartz, Yoram Singer,
            and Tushar Chandra.
            International Conference on Machine Learning (ICML 2008)
            http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf

        """
        v = CArray(v).ravel()
        d = v.size
        # check if we are already on the simplex
        if v.sum() == s and (v >= 0).sum() == d:
            return v  # best projection: itself!
        # get the array of cumulative sums of a sorted (decreasing) copy of v
        u = v.deepcopy()
        u.sort(inplace=True)
        u = u[::-1]
        if u.issparse:
            u_nnz = CArray(u.nnz_data).todense()
            cssv = u_nnz.cumsum()
        else:
            cssv = u.cumsum()

        # get the number of > 0 components of the optimal solution
        # (only considering non-null elements in v
        j = CArray.arange(1, cssv.size+1)
        if u.issparse:
            rho = (j * u_nnz > (cssv - s)).sum() - 1
        else:
            rho = (j * u > (cssv - s)).sum() - 1

        # compute the Lagrange multiplier associated to the simplex constraint
        theta = (cssv[rho] - s) / (rho + 1.0)

        # compute the projection by thresholding v using theta
        w = v
        if w.issparse:
            p = CArray(w.nnz_data).todense()
            p -= theta
            w[w.nnz_indices] = p
        else:
            w -= theta
        w[w < 0] = 0
        return w
