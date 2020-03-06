from secml.ml.classifiers.gradients.tests import \
    CClassifierGradientMixinTestCases

from secml.ml.classifiers import CClassifierSGD
from secml.array import CArray


class TestCClassifierGradientSGDMixin(CClassifierGradientMixinTestCases):
    """Unittests for CClassifierGradientSGDMixin."""

    def test_not_implemented(self):
        """Test `grad_tr_params`."""
        with self.assertRaises(NotImplementedError):
            CClassifierSGD('hinge', 'l2').grad_tr_params(
                CArray([]), CArray([]))


if __name__ == '__main__':
    CClassifierGradientMixinTestCases.main()
