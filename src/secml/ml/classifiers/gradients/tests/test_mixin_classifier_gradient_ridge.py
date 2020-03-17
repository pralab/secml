from secml.ml.classifiers.gradients.tests import \
    CClassifierGradientMixinTestCases
from secml.ml.classifiers.gradients.tests.test_classes import \
    CClassifierGradientTestRidge

from secml.ml.classifiers import CClassifierRidge
from secml.ml.features.normalization import CNormalizerMinMax


class TestCClassifierGradientRidgeMixin(CClassifierGradientMixinTestCases):
    """Unittests for CClassifierGradientRidgeMixin."""
    clf_grads_class = CClassifierGradientTestRidge()

    def test_grad_tr_params_linear(self):
        """Test `grad_tr_params` on a linear classifier."""

        for n in (None, CNormalizerMinMax((-10, 10))):
            clf = CClassifierRidge(preprocess=n)
            clf.fit(self.ds)
            self._test_grad_tr_params(clf)


if __name__ == '__main__':
    CClassifierGradientMixinTestCases.main()
