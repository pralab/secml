from secml.ml.classifiers.gradients.tests import \
    CClassifierGradientMixinTestCases
from secml.ml.classifiers.gradients.tests.test_classes import \
    CClassifierGradientTestSVM

from secml.ml.classifiers import CClassifierSVM
from secml.ml.features.normalization import CNormalizerMinMax


class TestCClassifierGradientSVMMixin(CClassifierGradientMixinTestCases):
    """Unittests for CClassifierGradientSVMMixin."""
    clf_grads_class = CClassifierGradientTestSVM()

    def test_grad_tr_params_linear(self):
        """Test `grad_tr_params` on a linear classifier."""

        for n in (None, CNormalizerMinMax((-10, 10))):
            clf = CClassifierSVM(store_dual_vars=True, preprocess=n)
            clf.fit(self.ds)
            self._test_grad_tr_params(clf)

    def test_grad_tr_params_nonlinear(self):
        """Test `grad_tr_params` on a nonlinear classifier."""

        for n in (None, CNormalizerMinMax((-10, 10))):
            clf = CClassifierSVM(kernel='rbf', preprocess=n)
            clf.fit(self.ds)
            self._test_grad_tr_params(clf)


if __name__ == '__main__':
    CClassifierGradientMixinTestCases.main()
