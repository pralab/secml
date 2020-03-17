from secml.adv.attacks.poisoning import CAttackPoisoningSVM
from secml.adv.attacks.poisoning.tests import CAttackPoisoningTestCases
from secml.ml.classifiers import CClassifierSVM
from secml.ml.features.normalization import CNormalizerMinMax


class TestCAttackPoisoningLinearSVM(CAttackPoisoningTestCases):
    """Unit test for CAttackPoisoningLinearSVM."""

    def setUp(self):
        clf_params = {'kernel': 'linear', 'C': 0.1}
        self._set_up(clf_idx='lin-svm',
                     poisoning_class=CAttackPoisoningSVM,
                     clf_class=CClassifierSVM,
                     clf_params=clf_params)

    def test_poisoning_with_normalization_inside(self):
        """Test the CAttackPoisoning object when the classifier contains a
        normalizer.
        """
        normalizer = CNormalizerMinMax(feature_range=(-10, 10))

        self._test_clf_accuracy(normalizer)

        # test if the attack is effective and eventually show 2D plots
        self._test_attack_effectiveness(normalizer)

    def test_poisoning_without_normalization_inside(self):
        """Test the CAttackPoisoning object when the classifier does not
        contains a normalizer.
        """

        self._test_clf_accuracy(normalizer=None)

        # test if the attack is effective and eventually show 2D plots
        self._test_attack_effectiveness(normalizer=None)


class TestCAttackPoisoningLinearRBF(CAttackPoisoningTestCases):
    """Unit test for CAttackPoisoningRBFSVM."""

    def setUp(self):
        clf_params = {'kernel': 'rbf', 'C': 10}
        self._set_up(clf_idx='RBF-svm',
                     poisoning_class=CAttackPoisoningSVM,
                     clf_class=CClassifierSVM,
                     clf_params=clf_params)

    def test_poisoning_with_normalization_inside(self):
        """Test the CAttackPoisoning object when the classifier contains a
        normalizer.
        """
        normalizer = CNormalizerMinMax(feature_range=(-10, 10))

        self._test_clf_accuracy(normalizer)

        # test if the attack is effective and eventually show 2D plots
        self._test_attack_effectiveness(normalizer)

    def test_poisoning_without_normalization_inside(self):
        """Test the CAttackPoisoning object when the classifier does not
        contains a normalizer.
        """

        self._test_clf_accuracy(normalizer=None)

        # test if the attack is effective and eventually show 2D plots
        self._test_attack_effectiveness(normalizer=None)


if __name__ == '__main__':
    CAttackPoisoningTestCases.main()
