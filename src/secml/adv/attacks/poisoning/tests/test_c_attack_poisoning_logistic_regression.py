from secml.adv.attacks.poisoning import CAttackPoisoningLogisticRegression
from secml.adv.attacks.poisoning.tests import CAttackPoisoningTestCases
from secml.ml.classifiers import CClassifierLogistic
from secml.ml.features.normalization import CNormalizerMinMax


class TestCAttackPoisoningLogisticRegression(CAttackPoisoningTestCases):
    """Unit test for CAttackPoisoningLogisticRegression."""

    def setUp(self):
        clf_params = {'C': 100, 'random_state': 42}
        self._set_up(clf_idx='logistic_regression',
                     poisoning_class=CAttackPoisoningLogisticRegression,
                     clf_class=CClassifierLogistic,
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
