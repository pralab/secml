from secml.utils import CUnitTest

from secml.core.constants import eps


class CExplainerLocalInfluenceTestCases(CUnitTest):
    """Unittests interface for CExplainerLocalInfluence."""

    def _test_gradient_numerical(self, clf, x, epsilon=eps, **kwargs):
        """Test for clf.gradient_f_x comparing to numerical gradient.

        Parameters
        ----------
        clf : CClassifier
        x : CArray

        """

        def _fun_args(sample, classifier, *f_args):
            return classifier.decision_function(sample, **f_args[0])

        def _grad_args(sample, classifier, *f_args):
            """
            Wrapper needed as gradient_f_x have **kwargs
            """
            return classifier.gradient_f_x(sample, **f_args[0])

        pass


if __name__ == '__main__':
    CUnitTest.main()
