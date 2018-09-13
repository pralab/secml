"""
Created on 18 feb 2016

@author: Davide Maiorca
"""
import unittest
from prlib.utils import CUnitTest

from prlib.optimization.c_optimizer import COptimizer
from prlib.optimization.constraints import CConstraintBox
from prlib.optimization.function import CFunction
from prlib.figure import CFigure
from prlib.array import CArray


class TestCOptimizer(CUnitTest):
    """Test for COptimizer class."""

    def test_minimize(self):
        """Testing function minimization."""
        self.logger.info("Test for minimize methods ... ")
        self.n_ft = 100
        A = CArray.eye(self.n_ft)
        # b = CArray.zeros(self.n_ft).T
        b = -0.1 * CArray.ones(self.n_ft).T
        self.circle = CFunction.create('quadratic', A, b, 0)
        startPoint = CArray.ones(self.n_ft)

        opt = []
        opt.append(COptimizer(solver='gradient', max_iter=2000, eta=0.1))
        opt.append(COptimizer(solver='descent_direction',
                              max_iter=2000,
                              eta=0.005, n_dimensions=100))

        box = CConstraintBox(lb=-1, ub=1)
        for i in range(len(opt)):
            with self.timer():
                # TODO: move inside optimizer/solver
                self.circle.reset()  # resets grad and fun evals
                self.logger.info("**** Testing Optimizer " + str(i) +
                                 " ****")
                opt[i].minimize(self.circle, startPoint, box)
                self.logger.info("Result point : " + str(opt[i].x_seq[-1, :]))
                self.logger.info("Number of fun/grad evaluations: " +
                                 str(self.circle.n_fun_eval) + "/" +
                                 str(self.circle.n_grad_eval))

        fig = CFigure(height=6, width=12)
        if self.n_ft is 2:

            grid_limits = [(-4, 4), (-4, 4)]
            for i in xrange(1, len(opt) + 1):
                fig.subplot(1, len(opt) + 1, i)

                fig.switch_sptype(sp_type='function')

                fig.sp.plot_fobj(func=CArray.apply_fun_torow, plot_levels=False,
                                 grid_limits=grid_limits, func_args=(self.circle.fun, ))

                fig.sp.plot_fobj(func=box.constraint, plot_background=False, levels=[0], n_grid_points=50,
                                 grid_limits=grid_limits)

                fig.sp.plot_path(opt[i - 1].x_seq)
                fig.sp.title("Path of optimizer " + str(i - 1))
            fig.subplot(1, len(opt) + 1, len(opt) + 1)
            for i in xrange(len(opt)):
                fig.sp.plot(opt[i].f_seq, label='opt ' + str(i))
            fig.sp.legend()
            fig.sp.title("Objective functions g(x)")
            fig.show()

        else:
            for i in xrange(len(opt)):
                fig.sp.plot(opt[i].f_seq, label='opt ' + str(i))
            fig.sp.legend()
            fig.sp.title("Objective functions g(x)")
            fig.show()


if __name__ == "__main__":
    unittest.main()
