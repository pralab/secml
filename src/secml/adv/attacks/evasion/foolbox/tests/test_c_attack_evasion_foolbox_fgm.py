from secml.adv.attacks.evasion.foolbox.tests.c_attack_evasion_foolbox_testcases import CAttackEvasionFoolboxTestCases

try:
    import foolbox
    import torch
except ImportError:
    CAttackEvasionFoolboxTestCases.importskip("foolbox")

from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_fgm_attack \
    import CFoolboxFGML1, CFoolboxFGML2, CFoolboxFGMLinf


class TestCAttackEvasionFoolboxFGML1(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxFGML1"""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxFGML1, self).setUp()
        self.attack_class = CFoolboxFGML1

        self.attack_params = {'random_start': False}

        self.has_targeted = False
        self.has_untargeted = True

    def test_run_untargeted(self):
        self._test_run_untargeted()

    def test_check_foolbox_equal_untargeted(self):
        self._test_check_foolbox_equal_untargeted()

    def test_shapes(self):
        self._test_shapes()


class TestCAttackEvasionFoolboxFGML2(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxFGML2"""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxFGML2, self).setUp()
        self.attack_class = CFoolboxFGML2

        self.attack_params = {'random_start': False}

        self.has_targeted = False
        self.has_untargeted = True

    def test_run_untargeted(self):
        self._test_run_untargeted()

    def test_check_foolbox_equal_untargeted(self):
        self._test_check_foolbox_equal_untargeted()

    def test_shapes(self):
        self._test_shapes()


class TestCAttackEvasionFoolboxFGMLinf(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxFGMLinf"""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxFGMLinf, self).setUp()
        self.attack_class = CFoolboxFGMLinf

        self.attack_params = {'random_start': False}

        self.has_targeted = False
        self.has_untargeted = True

    def test_run_untargeted(self):
        self._test_run_untargeted()

    def test_check_foolbox_equal_untargeted(self):
        self._test_check_foolbox_equal_untargeted()

    def test_shapes(self):
        self._test_shapes()
