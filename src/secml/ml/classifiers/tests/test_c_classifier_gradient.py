from secml.utils import CUnitTest

import random

from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierSVM
from secml.optimization import COptimizer
from secml.optimization.function import CFunction
from secml.ml.features.normalization import CNormalizerMinMax
from secml.data import CDataset

class TestCClassifierGradient(CUnitTest):

    def setUp(self):

        # generate synthetic data
        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1, random_state=1).load()

        self.dataset_sparse = self.dataset.tosparse()

        self.clfs = [CClassifierSVM()]
        self.clf_ids = ['lin-SVM']

        for clf in self.clfs:  # Enabling debug output for each classifier
            clf.verbose = 2

        self.logger.info("." * 50)
        self.logger.info("Number of Patterns: %s",
                         str(self.dataset.num_samples))
        self.logger.info("Features: %s", str(self.dataset.num_features))

    def _clf_gradient_check(self, clf):

        for i in random.sample(xrange(self.dataset.num_samples), 3):
            pattern = self.dataset.X[i, :]

            self.logger.info("P {:}: {:}".format(i, pattern))

            # Compare the analytical grad with the numerical grad
            gradient = clf.gradient_f_x(pattern, y=1)
            self.logger.info("Gradient: %s", str(gradient))
            check_grad_val = COptimizer(
                CFunction(clf.decision_function,
                          clf._gradient_f)).check_grad(pattern)
            self.logger.info(
                "norm(grad - num_grad): %s", str(check_grad_val))
            self.assertLess(check_grad_val, 1e-3,
                            "problematic kernel is " +
                            clf.kernel.class_type)
            for i, elm in enumerate(gradient):
                self.assertIsInstance(elm, float)

    def test_f_x_gradient(self):
        """Test the gradient of the classifier discriminant function"""
        self.logger.info("Testing the gradient of the discriminant function")

        for clf, clf_idx in zip(self.clfs, self.clf_ids):
            self.logger.info("Computing gradient for the classifier: %s when "
                             "the classifier does not have a normalizer "
                             "inside", clf_idx)

            clf.fit(self.dataset)
            self._clf_gradient_check(clf)

    def _test_f_norm_x_gradient(self):
        """Test the gradient of the classifier discriminant function
        when the classifier have a normalizer inside"""
        self.logger.info(
            "Testing the gradient of the discriminant function when the "
            "classifier have a normalizer inside")

        # create a normalizer
        normalizer = CNormalizerMinMax()

        for clf, clf_idx in zip(self.clfs, self.clf_ids):
            self.logger.info("Computing gradient for the classifier: %s when "
                             "the classifier have a normalizer "
                             "inside", clf_idx)

            clf.preprocess = normalizer
            clf.fit(self.dataset)
            self._clf_gradient_check(clf)

if __name__ == '__main__':
    CUnitTest.main()
