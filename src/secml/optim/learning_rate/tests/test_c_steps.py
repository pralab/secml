import unittest
from secml.utils import CUnitTest
from secml.array import CArray
from secml.figure import CFigure
from secml.utils import fm

from secml.optim.learning_rate import CStepConstant, CStepExponential, \
    CStepInvscaling, CStepOptimal


class TestCStep(CUnitTest):
    """Test for CStep subclasses."""

    def test_Compute_step(self):
        """Testing Step Classes."""
        self.logger.info(
            "Test for CStep subclasses... ")
        self.step_const = CStepConstant(1)
        self.step_exp = CStepExponential(0.95)
        self.step_invsc = CStepInvscaling(1, 2)
        self.step_opt = CStepOptimal(1, 2)

        self.filename = 'test_c_steps.pdf'

        x_range = CArray.arange(0, 10.1, 1)
        res_const = CArray.zeros(11)
        res_exp = CArray.zeros(11)
        res_invsc = CArray.zeros(11)
        res_opt = CArray.zeros(11)

        for i in xrange(0, 11):
            res_const[i] = self.step_const.get_actual_step(i)
            res_exp[i] = self.step_exp.get_actual_step(i)
            res_invsc[i] = self.step_invsc.get_actual_step(i)
            res_opt[i] = self.step_opt.get_actual_step(i)

        fig = CFigure(height=8, width=10)

        fig.subplot(2, 2, 1)
        fig.sp.plot(x_range, res_const)
        fig.sp.title("CStepConstant")
        fig.subplot(2, 2, 2)
        fig.sp.plot(x_range, res_exp)
        fig.sp.title("CStepExp")
        fig.subplot(2, 2, 3)
        fig.sp.plot(x_range, res_invsc)
        fig.sp.title("CStepInvscaling")
        fig.subplot(2, 2, 4)
        fig.sp.plot(x_range, res_opt)
        fig.sp.title("CStepOptimal")
        fig.savefig(fm.join(fm.abspath(__file__), self.filename),
                    file_format='pdf')


if __name__ == "__main__":
    unittest.main()
