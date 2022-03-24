from secml.adv.attacks.evasion.foolbox.tests.c_attack_evasion_foolbox_testcases import CAttackEvasionFoolboxTestCases

try:
    import foolbox
    import torch
except ImportError:
    CAttackEvasionFoolboxTestCases.importskip("foolbox")

from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_ead_attack import CFoolboxEAD


class TestCAttackEvasionFoolboxEAD(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxDDN"""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxEAD, self).setUp()
        self.attack_class = CFoolboxEAD

        self.attack_params = {'steps': self.default_steps, 'binary_search_steps': 9,
                              'confidence': 0.1, 'initial_stepsize': 1e-1,
                              'epsilons': None, 'abort_early': False}

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
