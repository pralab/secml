from secml.testing import CUnitTest
from secml.adv.attacks.evasion.tests.test_evasion_reject import \
    CEvasionRejectTestCases

from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.kernels import CKernelRBF


class TestEvasionRejectThreshold(CEvasionRejectTestCases.TestCEvasionReject):

    def _classifier_creation(self):
        # self.kernel = None
        self.kernel = CKernelRBF(gamma=1)

        self.multiclass = CClassifierMulticlassOVA(
            classifier=CClassifierSVM, class_weight='balanced',
            preprocess=None, kernel=self.kernel)
        self.multiclass.verbose = 0

        self.multiclass = CClassifierRejectThreshold(self.multiclass, 0.6)


if __name__ == '__main__':
    CUnitTest.main()
