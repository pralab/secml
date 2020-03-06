from secml.testing import CUnitTest

from secml.array import CArray
from secml.optim.function import CFunction
from secml.data.loader import CDLRandom


class CClassifierGradientMixinTestCases(CUnitTest):
    """Unittests interface for CClassifierGradientMixin.

    Attributes
    ----------
    clf_grads_class : CClassifierGradientTest
        Test class implementing gradient test methods for specific clf.

    """
    clf_grads_class = None

    @classmethod
    def setUpClass(cls):

        CUnitTest.setUpClass()

        cls.seed = 2

        cls.ds = CDLRandom(n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           random_state=cls.seed).load()
        cls.ds_sparse = cls.ds.tosparse()

    @staticmethod
    def _grad_tr_fun(params, x0, y0, clf_grads, clf):
        """Change classifier parameters and recompute objective function.

        Parameters
        ----------
        params : CArray
            New value of the parameters. Vector array.
        x0 : CArray
            Point on which to compute the objective function.
        y0 : CArray
            Label of the point on which to compute the objective function.
        clf_grads : CClassifierGradientTest
            Test class implementing gradient test methods.
        clf : CClassifier
            Target classifier.

        """
        clf = clf_grads.change_params(params, clf)
        return clf_grads.train_obj(x0, y0, clf)

    def _test_grad_tr_params(self, clf):
        """Compare `grad_tr_params` output with numerical gradient.

        Parameters
        ----------
        clf : CClassifier

        """
        i = self.ds.X.randsample(
            CArray.arange(self.ds.num_samples), 1, random_state=self.seed)
        x, y = self.ds.X[i, :], self.ds.Y[i]
        self.logger.info("P {:}: x {:}, y {:}".format(i.item(), x, y))

        params = self.clf_grads_class.params(clf)

        # Compare the analytical grad with the numerical grad
        gradient = clf.grad_tr_params(x, y).ravel()
        num_gradient = CFunction(self._grad_tr_fun).approx_fprime(
            params, epsilon=1e-6,
            x0=x, y0=y, clf_grads=self.clf_grads_class, clf=clf)

        error = (gradient - num_gradient).norm()

        self.logger.info("Analytical gradient:\n{:}".format(gradient))
        self.logger.info("Numerical gradient:\n{:}".format(num_gradient))

        self.logger.info("norm(grad - grad_num): {:}".format(error))
        self.assertLess(error, 1e-2)

        self.assertTrue(gradient.is_vector_like)
        self.assertEqual(params.size, gradient.size)
        self.assertEqual(params.issparse, gradient.issparse)
        self.assertIsSubDtype(gradient.dtype, float)
