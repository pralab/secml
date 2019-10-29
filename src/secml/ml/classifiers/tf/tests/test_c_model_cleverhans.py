from secml.testing import CUnitTest

try:
    import tensorflow as tf
except ImportError:
    CUnitTest.importskip("tensorflow")
else:
    import tensorflow as tf

from six.moves import range
from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.features.normalization import CNormalizerMinMax

from secml.ml.classifiers import CModelCleverhans


class TestCModelCleverhans(CUnitTest):
    """Unittests for CModelCleverhans."""

    def _dataset_creation(self):
        # generate synthetic data
        self.tr = CDLRandom(n_samples=100, n_classes=self.n_classes,
                            n_features=2, n_redundant=0,
                            n_clusters_per_class=1,
                            class_sep=1, random_state=0).load()

        # Add a new class modifying one of the existing clusters
        self.tr.Y[(self.tr.X[:, 0] > 0).logical_and(
            self.tr.X[:, 1] > 1).ravel()] = self.tr.num_classes

        self.lb = 0
        self.ub = 1

        # Data normalization
        self.normalizer = CNormalizerMinMax(
            feature_range=(self.lb, self.ub))
        self.normalizer = None
        if self.normalizer is not None:
            self.tr.X = self.normalizer.fit_normalize(self.tr.X)

    def setUp(self):
        self.n_classes = 3

        self._dataset_creation()

        # create a CClassifier
        self.clf = CClassifierMulticlassOVA(
            classifier=CClassifierSVM, class_weight='balanced',
            preprocess=None, kernel='linear')

        self.clf.fit(self.tr)

        # given a CClassifier, cree a CClassifierCleverhans
        self.clvh_clf = CModelCleverhans(
            self.clf, out_dims=self.n_classes)

        self._sess = tf.compat.v1.Session()

    def test_discriminant_function(self):

        self.logger.info("Check CClassifierCleverhans scores")

        # compute the score matrix with the CClassifier
        scores = self.clf.predict(
            self.tr.X, return_decision_function=True)[1]

        # compute the score matrix with the CClassifierCleverhans
        clvh_scores_tensor = self.clvh_clf.fprop(
            self.tr.X.tondarray())['logits']
        clvh_scores = CArray(self._sess.run(clvh_scores_tensor))

        self.assert_array_equal(scores.astype('float32'), clvh_scores)

    def test_gradient(self):
        """
        Firstly, we compute the gradient of the discriminant function w.r.t.
        all the
        classes (one at time) creating a matrix (n_classes * num_features)
        with a CClassifier. Then we do the same with a CCleverHans and we
        compare the results.
        """
        self.logger.info(
            "Check CClassifierCleverhans gradients")

        x = self.tr[0, :].X.ravel()

        # compute the gradient matrix with the CClassifier
        gradients = CArray.zeros((self.tr.num_classes, self.tr.num_features))
        for c in range(self.tr.num_classes):
            gradients[c, :] = self.clf.grad_f_x(x, y=c)
        gradients = gradients.tondarray()

        # compute the gradient matrix with the CClassifierCleverHans

        # convert the sample in a tensorflow constant
        x = tf.constant(x.ravel().tondarray())

        # get the tensor that contain the result of the discriminant function
        clvh_scores_tensor = self.clvh_clf._callable_fn(x)[0]

        clvh_gradients = CArray.zeros(
            (self.tr.num_classes, self.tr.num_features))
        for c in range(self.tr.num_classes):
            # for each class compute the gradient (selecting the class using
            # a one-hot vector)
            grad_init = CArray.zeros(shape=(self.tr.num_classes,))
            grad_init[c] = 1

            grads_tensor = tf.gradients(
                clvh_scores_tensor, [x], grad_ys=grad_init.tondarray())[0]
            grad = self._sess.run(grads_tensor)
            clvh_gradients[c, :] = CArray(grad)

        self.assert_array_equal(
            gradients.astype('float32'), clvh_gradients)


if __name__ == '__main__':
    CUnitTest.main()
