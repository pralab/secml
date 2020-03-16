from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.loss.c_softmax import CSoftmax
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.optim.function import CFunction
from secml.testing import CUnitTest


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

    def test_softmax_gradient(self):
        """Unittests for softmax gradient:
           Compare analytical gradients with its numerical approximation."""

        self.softmax = CSoftmax()

        def _sigma_pos_label(s, y):
            """
            Compute the sigmoid for the scores in s and return the i-th
            element of the vector that contains the results

            Parameters
            ----------
            s: CArray
                scores
            pos_label: index of the considered score into the vector

            Returns
            -------
            softmax: CArray
            """
            softmax = self.softmax.softmax(s).ravel()
            return softmax[y]

        score = self.scores[0, :]

        for pos_label in (0, 1, 2):
            self.logger.info("POS_LABEL: {:}".format(pos_label))

            # real value of the gradient on x
            grad = self.softmax.gradient(score, pos_label)

            self.logger.info("ANALITICAL GRAD: {:}".format(grad))

            approx = CFunction(_sigma_pos_label).approx_fprime(
                score, 1e-5, pos_label)

            self.logger.info("NUMERICAL GRADIENT: {:}".format(approx))

            check_grad_val = (grad - approx).norm()

            self.logger.info("The norm of the difference bettween the "
                             "analytical and the numerical gradient is: %s",
                             str(check_grad_val))
            self.assertLess(check_grad_val, 1e-4,
                            "the gradient is wrong {:}".format(check_grad_val))


if __name__ == '__main__':
    CUnitTest.main()
