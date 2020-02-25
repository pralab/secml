from secml.ml.classifiers.reject.tests import CClassifierRejectTestCases

from secml.ml.classifiers.reject import CClassifierRejectThreshold

from secml import _NoValue
from secml.ml.peval.metrics import CMetric
from secml.data.loader import CDLRandomBlobs
from secml.ml.classifiers import CClassifierSGD
from secml.ml.classifiers.loss import *
from secml.ml.classifiers.regularizer import *
from secml.figure import CFigure


class TestCClassifierRejectThreshold(CClassifierRejectTestCases):
    """Unit test for CClassifierRejectThreshold."""

    def setUp(self):
        """Test for init and fit methods."""
        # generate synthetic data
        self.dataset = CDLRandomBlobs(n_features=2, n_samples=100, centers=2,
                                      cluster_std=2.0, random_state=0).load()

        self.logger.info("Testing classifier creation ")
        self.clf_norej = CClassifierSGD(regularizer=CRegularizerL2(),
                                        loss=CLossHinge(), random_state=0)

        self.clf = CClassifierRejectThreshold(self.clf_norej, threshold=0.6)
        self.clf.verbose = 2  # Enabling debug output for each classifier
        self.clf.fit(self.dataset)

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        self.logger.info(
            "Test for decision_function() and predict() methods.")

        scores_d = self._test_fun(self.clf, self.dataset.todense())
        scores_s = self._test_fun(self.clf, self.dataset.tosparse())

        # FIXME: WHY THIS TEST IS CRASHING? RANDOM_STATE MAYBE?
        # self.assert_array_almost_equal(scores_d, scores_s)

    def test_reject(self):
        clf = self.clf_norej.deepcopy()
        clf_reject = self.clf.deepcopy()

        # Training the classifiers
        clf_reject.fit(self.dataset)
        clf.fit(self.dataset)

        # Classification of another dataset
        y_pred_reject, score_pred_reject = clf_reject.predict(
            self.dataset.X, n_jobs=_NoValue, return_decision_function=True)
        y_pred, score_pred = clf.predict(self.dataset.X,
                                         return_decision_function=True)

        # Compute the number of rejected samples
        n_rej = (y_pred_reject == -1).sum()
        self.logger.info("Rejected samples: {:}".format(n_rej))

        self.logger.info("Real: \n{:}".format(self.dataset.Y))
        self.logger.info("Predicted: \n{:}".format(y_pred))
        self.logger.info(
            "Predicted with reject: \n{:}".format(y_pred_reject))

        acc = CMetric.create('accuracy').performance_score(
            y_pred, self.dataset.Y)
        self.logger.info("Accuracy no rejection: {:}".format(acc))

        rej_acc = CMetric.create('accuracy').performance_score(
            y_pred_reject[y_pred_reject != -1],
            self.dataset.Y[y_pred_reject != -1])
        self.logger.info("Accuracy with rejection: {:}".format(rej_acc))

        # check that the accuracy using reject is higher that the one
        # without rejects
        self.assertGreaterEqual(
            rej_acc, acc, "The accuracy of the classifier that is allowed "
                          "to reject is lower than the one of the "
                          "classifier that is not allowed to reject")

    def test_gradient(self):
        """Unittest for gradient_f_x method."""

        i = 5  # Sample to test

        self.logger.info("Testing with dense data...")
        ds = self.dataset.todense()
        clf = self.clf.fit(ds)

        grads_d = self._test_gradient_numerical(
            clf, ds.X[i, :], extra_classes=[-1])

        self.logger.info("Testing with sparse data...")
        ds = self.dataset.tosparse()
        clf = self.clf.fit(ds)

        grads_s = self._test_gradient_numerical(
            clf, ds.X[i, :], extra_classes=[-1])

        # FIXME: WHY THIS TEST IS CRASHING? RANDOM_STATE MAYBE?
        # Compare dense gradients with sparse gradients
        # for grad_i, grad in enumerate(grads_d):
        #     self.assert_array_almost_equal(
        #         grad.atleast_2d(), grads_s[grad_i])

    def test_preprocess(self):
        """Test classifier with preprocessors inside."""
        # All linear transformations with gradient implemented
        self._test_preprocess(self.dataset, self.clf,
                              ['min-max', 'mean-std'],
                              [{'feature_range': (-1, 1)}, {}])
        self._test_preprocess_grad(self.dataset, self.clf,
                                   ['min-max', 'mean-std'],
                                   [{'feature_range': (-1, 1)}, {}],
                                   extra_classes=[-1])

        # Mixed linear/nonlinear transformations without gradient
        self._test_preprocess(
            self.dataset, self.clf, ['pca', 'unit-norm'], [{}, {}])

    def test_draw(self):
        """ Compare the classifiers graphically"""
        self.logger.info("Testing classifiers graphically")

        fig = CFigure(width=10, markersize=8)
        # Plot dataset points

        # mark the rejected samples
        y = self.clf.predict(self.dataset.X)
        fig.sp.plot_ds(
            self.dataset[y == -1, :], colors=['k', 'k'], markersize=12)

        # plot the dataset
        fig.sp.plot_ds(self.dataset)

        # Plot objective function
        fig.sp.plot_fun(self.clf.decision_function,
                        grid_limits=self.dataset.get_bounds(),
                        levels=[0], y=1)
        fig.sp.title('Classifier with reject threshold')

        fig.show()


if __name__ == '__main__':
    CClassifierRejectTestCases.main()
