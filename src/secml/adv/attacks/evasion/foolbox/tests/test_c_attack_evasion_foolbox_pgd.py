from secml.adv.attacks.evasion.foolbox.tests.c_attack_evasion_foolbox_testcases import CAttackEvasionFoolboxTestCases

try:
    import foolbox
    import torch
except ImportError:
    CAttackEvasionFoolboxTestCases.importskip("foolbox")

from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_pgd_attack \
    import CFoolboxPGDL1, CFoolboxPGDL2, CFoolboxPGDLinf


class TestCAttackEvasionFoolboxPGDL1(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxPGDL1"""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxPGDL1, self).setUp()
        self.attack_class = CFoolboxPGDL1

        self.attack_params = {'rel_stepsize': 0.025, 'steps': self.default_steps, 'abs_stepsize': 0.1,
                              'random_start': False}

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


class TestCAttackEvasionFoolboxPGDL2(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxPGDL2"""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxPGDL2, self).setUp()
        self.attack_class = CFoolboxPGDL2

        self.attack_params = {'rel_stepsize': 0.025, 'steps': self.default_steps, 'abs_stepsize': 0.1,
                              'random_start': False}

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


class TestCAttackEvasionFoolboxPGDLinf(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxPGDLinf"""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxPGDLinf, self).setUp()
        self.attack_class = CFoolboxPGDLinf

        self.attack_params = {'rel_stepsize': 0.025, 'steps': self.default_steps, 'abs_stepsize': 0.1,
                              'random_start': False}

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
