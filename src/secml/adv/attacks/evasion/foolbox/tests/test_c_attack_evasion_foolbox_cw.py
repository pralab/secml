from secml.adv.attacks.evasion.foolbox.tests.c_attack_evasion_foolbox_testcases import CAttackEvasionFoolboxTestCases

try:
    import foolbox
    import torch
except ImportError:
    CAttackEvasionFoolboxTestCases.importskip("foolbox")

from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_cw_attack import CFoolboxL2CarliniWagner


class TestCAttackEvasionFoolboxCW(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxCW"""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxCW, self).setUp()
        self.attack_class = CFoolboxL2CarliniWagner

        self.attack_params = {'steps': self.default_steps, 'abort_early': False}

        self.has_targeted = True
        self.has_untargeted = True

    def test_run_targeted(self):
        self._test_run_targeted()

    def test_run_untargeted(self):
        self._test_run_untargeted()

    def test_check_foolbox_equal_targeted(self):
        self._test_check_foolbox_equal_targeted()

    def test_check_foolbox_equal_untargeted(self):
        self._test_check_foolbox_equal_untargeted()

    def test_shapes(self):
        self._test_shapes()

    def test_obj_fun_and_grad(self):
        self._check_obj_function_and_grad()
