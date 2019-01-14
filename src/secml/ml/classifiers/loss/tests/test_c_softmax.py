from secml.utils import CUnitTest

from secml.ml.classifiers.loss import CSoftmax
from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.array import CArray


class TestCSoftmax(CUnitTest):
    """Unittests for CSoftmax."""

    def setUp(self):

        self.ds = CDLRandom(n_classes=3, n_samples=50, random_state=0,
                            n_informative=3).load()

        self.logger.info("Fit an SVM and classify dataset...")
        self.ova = CClassifierMulticlassOVA(CClassifierSVM)
        self.ova.fit(self.ds)
        self.labels, self.scores = self.ova.predict(
            self.ds.X, return_decision_function=True)

    def test_softmax(self):
        """Unittests for softmax function."""
        from sklearn.utils.extmath import softmax as softmax_sk

        sm = CSoftmax().softmax(self.scores)
        sm_sk = softmax_sk(self.scores.tondarray())

        self.logger.info("Our softmax.max():\n{:}".format(sm.max()))
        self.logger.info("SKlearn softmax.max():\n{:}".format(sm_sk.max()))

        self.assertFalse((sm.round(4) != CArray(sm_sk).round(4)).any())

        self.logger.info("Testing a single point...")

        sm = CSoftmax().softmax(self.scores[0, :])
        sm_sk = softmax_sk(self.scores[0, :].tondarray())

        self.logger.info("Our softmax.max():\n{:}".format(sm.max()))
        self.logger.info("SKlearn softmax.max():\n{:}".format(sm_sk.max()))

        self.assertFalse((sm.round(4) != CArray(sm_sk).round(4)).any())


if __name__ == '__main__':
    CUnitTest.main()
