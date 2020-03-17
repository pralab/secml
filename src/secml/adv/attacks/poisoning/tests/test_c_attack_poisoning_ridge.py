from secml.adv.attacks.poisoning import CAttackPoisoningRidge
from secml.adv.attacks.poisoning.tests import CAttackPoisoningTestCases
from secml.ml.classifiers import CClassifierRidge
from secml.ml.features.normalization import CNormalizerMinMax


class TestCAttackPoisoningRidge(CAttackPoisoningTestCases):
    """Unit test for CAttackPoisoningRidge."""

    def setUp(self):
        clf_params = {'fit_intercept': True, 'alpha': 1}
        self._set_up(clf_idx='ridge',
                     poisoning_class=CAttackPoisoningRidge,
                     clf_class=CClassifierRidge,
                     clf_params=clf_params)

    def test_poisoning_with_normalization_inside(self):
        """Test the CAttackPoisoning object when the classifier contains a
        normalizer.
        """
        normalizer = CNormalizerMinMax(feature_range=(-10, 10))

        self._test_clf_accuracy(normalizer)

        # test if the attack is effective and eventually show 2D plots
        self._test_attack_effectiveness(normalizer)

        self._test_single_poisoning_grad_check(normalizer)

    def test_poisoning_without_normalization_inside(self):
        """Test the CAttackPoisoning object when the classifier does not
        contains a normalizer.
        """

        self._test_clf_accuracy(normalizer=None)

        # test if the attack is effective and eventually show 2D plots
        self._test_attack_effectiveness(normalizer=None)

        self._test_single_poisoning_grad_check(normalizer=None)


if __name__ == '__main__':
    CAttackPoisoningTestCases.main()
