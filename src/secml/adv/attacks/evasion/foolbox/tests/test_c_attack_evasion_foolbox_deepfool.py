from secml.adv.attacks.evasion.foolbox.tests.c_attack_evasion_foolbox_testcases import CAttackEvasionFoolboxTestCases

try:
    import foolbox
    import torch
except ImportError:
    CAttackEvasionFoolboxTestCases.importskip("foolbox")

from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_deepfool_attack \
    import CFoolboxDeepfoolL2, CFoolboxDeepfoolLinf


class TestCAttackEvasionFoolboxDeepfoolL2Logits(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxDeepfoolL2 with difference of logits loss."""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxDeepfoolL2Logits, self).setUp()
        self.attack_class = CFoolboxDeepfoolL2

        self.attack_params = {'steps': 25, 'epsilons': None,
                              'loss': 'logits', 'candidates': 2,
                              'overshoot': 0.01}
        self.has_targeted = False
        self.has_untargeted = True

    def test_run_untargeted(self):
        self._test_run_untargeted()

    def test_check_foolbox_equal_untargeted(self):
        self._test_check_foolbox_equal_untargeted()

    def test_shapes(self):
        self._test_shapes()

    def test_obj_fun_and_grad(self):
        self._check_obj_function_and_grad()


class TestCAttackEvasionFoolboxDeepfoolLInfLogits(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxDeepfoolLInf with difference of logits loss."""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxDeepfoolLInfLogits, self).setUp()
        self.attack_class = CFoolboxDeepfoolLinf

        self.attack_params = {'steps': 100, 'epsilons': None,
                              'loss': 'logits', 'candidates': 2,
                              'overshoot': 0.01}
        self.has_targeted = False
        self.has_untargeted = True

    def test_run_untargeted(self):
        self._test_run_untargeted()

    def test_check_foolbox_equal_untargeted(self):
        self._test_check_foolbox_equal_untargeted()

    def test_shapes(self):
        self._test_shapes()

    def test_obj_fun_and_grad(self):
        self._check_obj_function_and_grad()


class TestCAttackEvasionFoolboxDeepfoolL2CELoss(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxDeepfoolL2 with difference of cross-entropies."""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxDeepfoolL2CELoss, self).setUp()
        self.attack_class = CFoolboxDeepfoolL2

        self.attack_params = {'steps': 100, 'epsilons': None,
                              'loss': 'crossentropy', 'candidates': 2,
                              'overshoot': 0.01}
        self.has_targeted = False
        self.has_untargeted = True

    def test_run_untargeted(self):
        self._test_run_untargeted()

    def test_check_foolbox_equal_untargeted(self):
        self._test_check_foolbox_equal_untargeted()

    def test_shapes(self):
        self._test_shapes()

    def test_obj_fun_and_grad(self):
        self._check_obj_function_and_grad()

class TestCAttackEvasionFoolboxDeepfoolLInfCELoss(CAttackEvasionFoolboxTestCases):
    """Unit test for CAttackEvasionFoolboxDeepfoolLInf with difference of cross-entropies."""

    make_figures = False  # Set as True to produce figures

    def setUp(self):
        super(TestCAttackEvasionFoolboxDeepfoolLInfCELoss, self).setUp()
        self.attack_class = CFoolboxDeepfoolLinf

        self.attack_params = {'steps': self.default_steps, 'epsilons': None,
                              'loss': 'crossentropy', 'candidates': 2,
                              'overshoot': 0.01}
        self.has_targeted = False
        self.has_untargeted = True

    def test_run_untargeted(self):
        self._test_run_untargeted()

    def test_check_foolbox_equal_untargeted(self):
        self._test_check_foolbox_equal_untargeted()

    def test_shapes(self):
        self._test_shapes()

    def test_obj_fun_and_grad(self):
        self._check_obj_function_and_grad()
