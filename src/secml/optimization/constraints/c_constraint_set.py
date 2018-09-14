'''

This class handles multiple constraints.

Last revision on 26/feb/2016
@author: Battista Biggio
'''

from secml.optimization.constraints.c_constraint import CConstraint


class CConstraintSet(CConstraint):

    class_type = "constraint-set"

    # TODO add checks on inputs
    def __init__(self, *args):
        # this should be a tuple/list of constraints
        # we need input validation on args
        self._constraints = args
        return

    # TODO: input checking: single pattern, output is scalar value
    def _constraint(self, x):
        '''
        Evaluates all constraints in the set
        and returns the highest value, namely,
        the first one which is likely to be violated.
        Recall that constraints have the form
            c(x) <= 0
        so, we basically return
            max_i c_i(x)
        '''
        score_max = self._constraints[0]._constraint(x)
        for constr in self._constraints:
            score = constr._constraint(x)
            if score >= score_max:
                score_max = score
        return float(score_max)

    # TODO: not guaranteed to work!
    def _projection(self, x):
        '''
        Projection on feasible domain.

        So far, we just project on each constraint sequentially.
        This does not guarantees however to get a feasible point.
        '''
        for constr in self._constraints:
            x = constr.projection(x)

        return x
