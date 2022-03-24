from secml.adv.attacks.evasion.tests import CAttackEvasionTestCases

try:
    import foolbox
    import torch
except ImportError:
    CAttackEvasionTestCases.importskip("foolbox")

from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor
from secml.array import CArray
import foolbox as fb
import torch


class CAttackEvasionFoolboxTestCases(CAttackEvasionTestCases):
    def setUp(self):
        ds, clf = self._prepare_nonlinear_svm(sparse=False, seed=0)
        self.ds = ds
        self.clf = clf
        self.clf.fit(self.ds.X, ds.Y)
        pt = 0
        self.x0, self.y0 = self.ds[pt, :].X, self.ds[pt, :].Y
        self.lb, self.ub = self.ds.X.min(), self.ds.X.max()

        self.default_steps = 15

        self.attack_class = None

        # those are required for initializing and running the foolbox attack
        self.attack_params = dict()

        # redefine these for running the tests in each of the cases
        self.has_targeted = False
        self.has_untargeted = False

    def _setup_attack(self, targeted=False):
        if targeted is True:
            y_target = 1 if self.y0.item() != 1 else 0
        else:
            y_target = None
        evas = self.attack_class(
            classifier=self.clf,
            y_target=y_target,
            lb=self.lb, ub=self.ub,
            **self.attack_params
        )
        return evas

    def _test_run_targeted(self):
        if self.has_targeted:
            evas = self._setup_attack(targeted=True)
            self._run_evasion(evas, self.x0, self.y0)
            self._plot_2d_evasion(evas, self.ds, self.x0,
                                  filename="{}_target_{}.pdf"
                                           "".format(self.attack_class.__name__,
                                                     evas.y_target))
        else:
            self.logger.debug("Targeted version not defined for {}, skipping test"
                              "".format(self.attack_class.__name__))
            return

    def _test_run_untargeted(self):
        if self.has_untargeted:
            evas = self._setup_attack(targeted=False)
            self._run_evasion(evas, self.x0, self.y0)
            self._plot_2d_evasion(evas, self.ds, self.x0, "{}_target_{}.pdf"
                                                          "".format(self.attack_class.__name__, evas.y_target))
        else:
            self.logger.debug("Untargeted version not defined for {}, skipping test"
                              "".format(self.attack_class.__name__))
            return

    def _test_check_foolbox_equal_targeted(self):
        if self.has_targeted:
            evas = self._setup_attack(targeted=True)
            foolbox_class = evas.attack_class
            init_params = self.attack_params
            if 'epsilons' in init_params:
                init_params.pop('epsilons')
            fb_evas = foolbox_class(**init_params)
            adv_ds, adv_fb = self._check_adv_example(evas, fb_evas)
            self.assert_array_almost_equal(adv_ds.X, adv_fb, decimal=3)
        else:
            self.logger.debug("Targeted version not defined for {}, skipping test"
                              "".format(self.attack_class.__name__))
            return

    def _test_check_foolbox_equal_untargeted(self):
        if self.has_untargeted:
            evas = self._setup_attack(targeted=False)
            foolbox_class = evas.attack_class
            init_params = self.attack_params
            if 'epsilons' in init_params:
                init_params.pop('epsilons')
            fb_evas = foolbox_class(**init_params)
            adv_ds, adv_fb = self._check_adv_example(evas, fb_evas)
            self.assert_array_almost_equal(adv_ds.X, adv_fb, decimal=3)
        else:
            self.logger.debug("Untargeted version not defined for {}, skipping test"
                              "".format(self.attack_class.__name__))
            return

    def _test_shapes(self):
        if self.has_untargeted:
            evas = self._setup_attack(targeted=False)
        elif self.has_targeted:
            evas = self._setup_attack(targeted=False)
        else:
            self.logger.debug("Nor targeted or untargeted versions are defined. Skipping test.")
        y_pred, scores, adv_ds, f_obj = evas.run(self.x0, self.y0)
        self.assert_array_equal(self.x0.shape, adv_ds.X.shape)
        self.assert_array_equal(self.y0.shape, adv_ds.Y.shape)
        path = evas.x_seq
        self.assertEqual(path.shape[1], self.x0.shape[1])

    def _check_adv_example(self, secml_attack, fb_attack):
        x0_tensor = as_tensor(self.x0.atleast_2d())
        y0_tensor = as_tensor(self.y0.ravel(), tensor_type=torch.LongTensor)

        y_target = secml_attack.y_target

        if y_target is None:
            criterion = fb.criteria.Misclassification(y0_tensor)
        else:
            criterion = fb.criteria.TargetedMisclassification(torch.tensor([y_target]))

        y_pred, scores, adv_ds, f_obj = secml_attack.run(self.x0, self.y0)
        _, adv_fb, _ = fb_attack(secml_attack.f_model, x0_tensor, criterion, epsilons=secml_attack.epsilon)
        adv_fb = CArray(adv_fb.numpy())
        return adv_ds, adv_fb

    def _check_obj_function_and_grad(self):
        for is_targeted, check in zip((True, False), (self.has_targeted, self.has_untargeted)):
            if check is True:
                evas = self._setup_attack(targeted=is_targeted)
                # some attacks require to run the attack before computing
                # the loss function
                _ = evas.run(self.x0, self.y0)
                obj_function = evas.objective_function(self.x0)
                obj_function_grad = evas.objective_function_gradient(self.x0)
                self.assertEqual(obj_function.shape, (self.x0.shape[0],))
                self.assertEqual(obj_function_grad.shape, self.x0.shape)
        return

