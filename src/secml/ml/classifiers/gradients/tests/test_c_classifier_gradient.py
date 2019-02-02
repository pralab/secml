from secml.utils import CUnitTest
from abc import ABCMeta, abstractmethod
from secml.array import CArray
from secml.ml.features.normalization import CNormalizerMinMax
from secml.optimization import COptimizer
from secml.optimization.function import CFunction

# fixme: add the test for the other gradient. as not all the classifier will
#  be implemented we can check if the CCgradient class have that gradient to
#  understand if we should or not execute that test function
class CClassifierGradientTestCases(object):
    """Wrapper for TestCClassifierGradient to make unittest.main() work correctly."""

    class TestCClassifierGradient(CUnitTest):
        """Unit test for the classifier gradients."""
        __metaclass__ = ABCMeta

        @abstractmethod
        def _dataset_creation(self):
            raise NotImplementedError

        @abstractmethod
        def _clfs_creation(self):
            raise NotImplementedError

        def setUp(self):

            self.seed = 0

            self._dataset_creation()
            self._set_tested_classes()
            self._clfs_creation()

            self.logger.info("." * 50)
            self.logger.info("Number of Patterns: %s",
                             str(self.dataset.num_samples))
            self.logger.info("Features: %s", str(self.dataset.num_features))

        def _fun_args(self, x, *args):
            return self.clf.decision_function(x, **args[0])

        def _grad_args(self, x, *args):
            """
            Wrapper needed as gradient_f_x have **kwargs
            """
            return self.clf.gradient_f_x(x, **args[0])

        def _clf_gradient_check(self, clf, clf_idx):

            self.clf = clf

            smpls_idx = CArray.arange(self.dataset.num_samples)
            i = self.dataset.X.randsample(smpls_idx, 1, random_state=self.seed)
            pattern = self.dataset.X[i, :]
            self.logger.info("P {:}: {:}".format(i, pattern))

            for c in self.classes:

                # Compare the analytical grad with the numerical grad
                gradient = clf.gradient_f_x(pattern, y=c)
                num_gradient = COptimizer(
                    CFunction(self._fun_args,
                              self._grad_args)).approx_fprime(pattern, 1e-8,
                                                              ({'y': c}))
                error = (gradient - num_gradient).norm(order=1)
                self.logger.info("Analitic gradient w.r.t. class %s: %s",
                                 str(c), str(gradient))
                self.logger.info("Numeric gradient w.r.t. class %s: %s",
                                 str(c), str(num_gradient))

                self.logger.info(
                    "norm(grad - num_grad): %s", str(error))
                self.assertLess(error, 1e-3,"problematic classifier is " +
                                clf_idx)

                for i, elm in enumerate(gradient):
                    self.assertIsInstance(elm, float)

        def test_f_x_gradient(self):
            """Test the gradient of the classifier discriminant function"""
            self.logger.info(
                "Testing the gradient of the discriminant function")

            for clf, clf_idx in zip(self.clfs, self.clf_ids):
                self.logger.info(
                    "Computing gradient for the classifier: %s when "
                    "the classifier does not have a normalizer "
                    "inside", clf_idx)

                clf.fit(self.dataset)
                self._clf_gradient_check(clf, clf_idx)

        def test_f_norm_x_gradient(self):
            """Test the gradient of the classifier discriminant function
            when the classifier have a normalizer inside"""
            self.logger.info(
                "Testing the gradient of the discriminant function when the "
                "classifier have a normalizer inside")

            # create a normalizer
            normalizer = CNormalizerMinMax(feature_range=(-100,100))

            for clf, clf_idx in zip(self.clfs, self.clf_ids):
                self.logger.info(
                    "Computing gradient for the classifier: %s when "
                    "the classifier have a normalizer "
                    "inside", clf_idx)

                clf.preprocess = normalizer
                clf.fit(self.dataset)
                self._clf_gradient_check(clf, clf_idx)
