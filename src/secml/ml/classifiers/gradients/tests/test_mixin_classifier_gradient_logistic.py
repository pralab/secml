from secml.ml.classifiers.gradients.tests import \
    CClassifierGradientMixinTestCases
from secml.ml.classifiers.gradients.tests.test_classes import \
    CClassifierGradientTestLogisticRegression

from secml.ml.classifiers import CClassifierLogistic
from secml.ml.features.normalization import CNormalizerMinMax


class TestCClassifierGradientLogisticMixin(CClassifierGradientMixinTestCases):
    """Unittests for CClassifierGradientLogisticMixin."""
    clf_grads_class = CClassifierGradientTestLogisticRegression()

    def test_grad_tr_params_linear(self):
        """Test `grad_tr_params` on a linear classifier."""

        for n in (None, CNormalizerMinMax((-10, 10))):
            clf = CClassifierLogistic(preprocess=n)
            clf.fit(self.ds)
            self._test_grad_tr_params(clf)


if __name__ == '__main__':
    CClassifierGradientMixinTestCases.main()
