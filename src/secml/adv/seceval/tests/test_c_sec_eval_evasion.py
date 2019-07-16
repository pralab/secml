"""
Created on 08 mar 2016

@author: Davide Maiorca
@author: Ambra Demontis

This module tests the CSecEval class, which performs an attacks on
a dataset w.r.t increasing attack power
TODO: Add assertEqual statements to check the correctness of the test
"""
from secml.testing import CUnitTest
from secml.adv.seceval.tests import CSecEvalTestCases

from secml.adv.attacks.evasion import CAttackEvasionBLS
from secml.array import CArray


class TestCSecEvalEvasion(CSecEvalTestCases.TestCSecEval):
    """
    SecEvalEvasion unittest
    """

    def attack_params_setter(self):
        """
        Set evasion method dependent parameters
        :return:
        """

        # only manipulate positive samples, targeting negative ones
        self.y_target = None
        self.attack_classes = CArray([1])

        params = {
            "classifier": self.classifier,
            "surrogate_classifier": self.classifier,
            "surrogate_data": self.tr,
            "distance": 'l1',
            "lb": self.lb,
            "ub": self.ub,
            "discrete": False,
            # "eta": 1.0,
            #"eta_min": 120.0,
            #"eta_max": None,
            "y_target": self.y_target,
            "attack_classes": self.attack_classes
        }
        self.attack = CAttackEvasionBLS(**params)
        self.attack.verbose = 1

        # sec eval params
        self.param_name = 'dmax'
        dmax = 2
        dmax_step = 0.25
        self.param_values = CArray.arange(
            start=0, step=dmax_step,
            stop=dmax + dmax_step)


if __name__ == '__main__':
    CUnitTest.main()
