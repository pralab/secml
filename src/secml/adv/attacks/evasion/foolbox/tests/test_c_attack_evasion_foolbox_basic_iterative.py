from secml.adv.attacks.evasion.foolbox.tests.c_attack_evasion_foolbox_testcases import CAttackEvasionFoolboxTestCases

try:
    import foolbox
    import torch
except ImportError:
    CAttackEvasionFoolboxTestCases.importskip("foolbox")

from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_basic_iterative_attack \
    import CFoolboxBasicIterativeL1, \
    CFoolboxBasicIterativeL2, CFoolboxBasicIterativeLinf


class TestCAttackEvasionFoolboxBasicIterativeL1(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxBasicIterativeL1"""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxBasicIterativeL1, self).setUp()
        self.attack_class = CFoolboxBasicIterativeL1

        self.attack_params = {'rel_stepsize': 0.03, 'steps': 25, 'abs_stepsize': 0.1, 'random_start': False}

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


class TestCAttackEvasionFoolboxBasicIterativeL2(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxBasicIterativeL2"""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxBasicIterativeL2, self).setUp()
        self.attack_class = CFoolboxBasicIterativeL2

        self.attack_params = {'rel_stepsize': 0.03, 'steps': 100, 'abs_stepsize': 0.1, 'random_start': False}

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


class TestCAttackEvasionFoolboxBasicIterativeLinf(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxBasicIterativeLinf"""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxBasicIterativeLinf, self).setUp()
        self.attack_class = CFoolboxBasicIterativeLinf

        self.attack_params = {'rel_stepsize': 0.03, 'steps': self.default_steps, 'abs_stepsize': 0.1,
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