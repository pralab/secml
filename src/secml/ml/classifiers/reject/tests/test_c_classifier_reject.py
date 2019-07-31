from abc import ABCMeta
import six

from secml.ml.classifiers.tests import CClassifierTestCases

from secml import _NoValue
from secml.array import CArray
from secml.figure import CFigure
from secml.ml.peval.metrics import CMetric


class CClassifierRejectTestCases(object):

    @six.add_metaclass(ABCMeta)
    class TestCClassifierReject(CClassifierTestCases):
        """Unit test for CClassifierReject"""

        def _check_classify_scores(self, l, s, n_samples, n_classes):
            # Override as reject classifiers add one additional class
            self.assertEqual(type(l), CArray)
            self.assertEqual(type(s), CArray)
            self.assertTrue(l.isdense)
            self.assertTrue(s.isdense)
            self.assertEqual(1, l.ndim)
            self.assertEqual(2, s.ndim)
            self.assertEqual((n_samples,), l.shape)
            self.assertEqual((n_samples, n_classes + 1), s.shape)
            self.assertEqual(int, l.dtype)
            self.assertEqual(float, s.dtype)

        def test_fun(self):
            """Test for decision_function() and predict() methods."""
            self.logger.info(
                "Test for decision_function() and predict() methods.")

            self._test_fun(self.clf, self.dataset.todense())
            self._test_fun(self.clf, self.dataset.tosparse())

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
                "Predicted \w Reject: \n{:}".format(y_pred_reject))

            acc = CMetric.create('accuracy').performance_score(
                y_pred, self.dataset.Y)
            self.logger.info("Accuracy no rejection: {:}".format(acc))

            rej_acc = CMetric.create('accuracy').performance_score(
                y_pred_reject[y_pred_reject != -1],
                self.dataset.Y[y_pred_reject != -1])
            self.logger.info("Accuracy WITH rejection: {:}".format(rej_acc))

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
