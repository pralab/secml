"""
.. module:: CConstraintL1
   :synopsis: L1 Constraint

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.array import CArray
from secml.optim.constraints import CConstraint


class CConstraintL1(CConstraint):
    """L1 Constraint.

    Parameters
    ----------
    center : scalar or CArray, optional
        Center of the constraint. Use an array to specify a different
        value for each dimension. Default 0.
    radius : scalar, optional
        The semidiagonal of the constraint. Default 1.

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
        """Center of the constraint."""
        return self._center

    @center.setter
    def center(self, value):
        """Center of the constraint."""
        self._center = value

    @property
    def radius(self):
        """Semidiagonal of the constraint."""
        return self._radius

    @radius.setter
    def radius(self, value):
        """Semidiagonal of the constraint."""
        self._radius = float(value)

    def _constraint(self, x):
        """Returns the value of the constraint for the sample x.

        The constraint value y is given by:
         y = ||x - center||_1 - radius

        Parameters
        ----------
        x : CArray
            Input array.

        Returns
        -------
        float
            Value of the constraint.

        """
        return float((x - self.center).norm(order=1) - self.radius)

    def _projection(self, x):
        """Project x onto feasible domain / within the given constraint.

        Solves the optimisation problem (using the algorithm from [1]):
            min_w 0.5 * || w - x ||_2^2 , s.t. || w ||_1 <= s

        Parameters
        ----------
        x : CArray
            Input sample.

        Returns
        -------
        CArray
            Projected x onto feasible domain if constraint is violated.

        Notes
        -----
        Solves the problem by a reduction to the positive simplex case.

        """
        s = float(self.radius)
        v = (x - self.center).ravel()
        # compute the vector of absolute values
        u = abs(v)
        # check if v is already a solution
        if u.sum() <= s:
            # l1-norm is <= s
            out = v + self._center
            return out.tosparse() if x.issparse else out

        # v is not already a solution: optimum lies on the boundary (norm == s)
        # project *u* on the simplex
        w = self._euclidean_proj_simplex(u, s=s)
        # compute the solution to the original problem on v
        w *= v.sign()
        out = w + self._center
        return out.tosparse() if x.issparse else out

    def _euclidean_proj_simplex(self, v, s=1):
        """Compute the Euclidean projection on a positive simplex.

        Solves the optimisation problem (using the algorithm from [1]):

            min_w 0.5 * || w - v ||_2^2 ,
            s.t. \\sum_i w_i = s, w_i >= 0

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
        w = v.deepcopy()
        if w.issparse:
            p = CArray(w.nnz_data)
            p -= theta
            w = w.astype(p)  # p dtype may change after subtraction
            w[w.nnz_indices] = p
        else:
            w -= theta
        w[w < 0] = 0
        return w

    def _gradient(self, x):
        """Returns the gradient of c(x) in x.

        Parameters
        ----------
        x : CArray
            Input sample.

        Returns
        -------
        CArray
            The gradient of the constraint computed on x.

        """
        return (x - self.center).sign().ravel()
