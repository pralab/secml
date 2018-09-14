"""
Created on 08 mar 2016

@author: Davide Maiorca
@author: Ambra Demontis

This module tests the CSecEval class, which performs an attacks on
a dataset w.r.t increasing attack power
TODO: Add assertEquals statements to check the correctness of the test
"""
import unittest
from secml.array import CArray

from secml.adv.attacks.evasion import CAttackEvasion
from test_c_sec_eval import TestCSecEvalCases


class TestCSecEvalEvasion(TestCSecEvalCases.TestCSecEval):
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
        self.attack_classes = 1

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
        self.attack = CAttackEvasion(**params)
        self.attack.verbose = 1

        # sec eval params
        self.param_name = 'dmax'
        dmax = 10 # 10000
        dmax_step = 1 # 1000
        self.param_values = CArray.arange(
            start=0, step=dmax_step,
            stop=dmax + 1)


if __name__ == '__main__':
    unittest.main()
