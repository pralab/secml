from abc import ABCMeta, abstractmethod

from secml.ml.classifiers.tests import CClassifierTestCases

from secml import _NoValue
from secml.array import CArray
from secml.figure import CFigure
from secml.ml.peval.metrics import CMetric


class CClassifierRejectTestCases(object):

    class TestCClassifierReject(CClassifierTestCases):
        """Unit test for CClassifierReject"""
        __metaclass__ = ABCMeta

        @abstractmethod
        def setUp(self):
            raise NotImplementedError()

        # fixme: move this plot on an example file
        def test_draw(self):
            """ Compare the classifiers graphically"""
            self.logger.info("Testing classifiers graphically")

            # generate 2D synthetic data
            dataset = self.dataset

            clf = self.clf.deepcopy()
            clf.fit(dataset)

            fig = CFigure(width=10, markersize=8)
            # Plot dataset points
            fig.switch_sptype(sp_type='ds')

            # mark the rejected samples
            y = clf.predict(dataset.X)
            fig.sp.plot_ds(
                dataset[y == -1, :], colors=['k', 'k'], markersize=12)

            # plot the dataset
            fig.sp.plot_ds(dataset)

            # Plot objective function
            fig.switch_sptype(sp_type='function')
            fig.sp.plot_fobj(clf.decision_function,
                             grid_limits=dataset.get_bounds(), y=1)
            fig.sp.title('Classifier with reject threshold')

            fig.show()

        def test_fun(self):
            """Test for decision_function() and predict() methods."""
            self.logger.info(
                "Test for decision_function() and predict() methods.")

            def _check_df_scores(s, n_samples):
                self.assertEqual(type(s), CArray)
                self.assertTrue(s.isdense)
                self.assertEqual(1, s.ndim)
                self.assertEqual((n_samples,), s.shape)
                self.assertEqual(float, s.dtype)

            def _check_classify_scores(l, s, n_samples, n_classes):
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

            x = x_norm = self.dataset.X
            p = p_norm = self.dataset.X[0, :].ravel()

            # Preprocessing data if a preprocess is defined
            if self.clf.preprocess is not None:
                x_norm = self.clf.preprocess.normalize(x)
                p_norm = self.clf.preprocess.normalize(p)

            # Testing decision_function on multiple points

            df_scores_nerej = self.clf.decision_function(x, y=-1)
            self.logger.info("decision_function(x, y=-1):\n"
                             "{:}".format(df_scores_nerej))
            _check_df_scores(df_scores_nerej, self.dataset.num_samples)

            df_scores_neg = self.clf.decision_function(x, y=0)
            self.logger.info("decision_function(x, y=0):\n"
                             "{:}".format(df_scores_neg))
            _check_df_scores(df_scores_neg, self.dataset.num_samples)

            df_scores_pos = self.clf.decision_function(x, y=1)
            self.logger.info("decision_function(x, y=1):\n"
                             "{:}".format(df_scores_pos))
            _check_df_scores(df_scores_pos, self.dataset.num_samples)

            # Testing _decision_function on multiple points

            ds_priv_scores = self.clf._decision_function(x_norm, y=1)
            self.logger.info("_decision_function(x_norm, y=1):\n"
                             "{:}".format(ds_priv_scores))
            _check_df_scores(ds_priv_scores, self.dataset.num_samples)

            # Comparing output of public and private

            self.assertFalse((df_scores_pos != ds_priv_scores).any())

            # Testing predict on multiple points

            labels, scores = self.clf.predict(x, return_decision_function=True,
                                              n_jobs=_NoValue)
            self.logger.info("predict(x):\nlabels: {:}\n"
                             "scores: {:}".format(labels, scores))
            _check_classify_scores(
                labels, scores, self.dataset.num_samples, self.clf.n_classes)

            # Comparing output of decision_function and predict

            self.assertFalse((df_scores_neg != scores[:, 0].ravel()).any())
            self.assertFalse((df_scores_pos != scores[:, 1].ravel()).any())

            # Testing decision_function on single point

            df_scores_neg = self.clf.decision_function(p, y=0)
            self.logger.info("decision_function(p, y=0):\n"
                             "{:}".format(df_scores_neg))
            _check_df_scores(df_scores_neg, 1)

            df_scores_pos = self.clf.decision_function(p, y=1)
            self.logger.info("decision_function(p, y=1):\n"
                             "{:}".format(df_scores_pos))
            _check_df_scores(df_scores_pos, 1)

            # Testing _decision_function on single point

            df_priv_scores = self.clf._decision_function(p_norm, y=1)
            self.logger.info("_decision_function(p_norm, y=1):\n"
                             "{:}".format(df_priv_scores))
            _check_df_scores(df_priv_scores, 1)

            # Comparing output of public and private

            self.assertFalse((df_scores_pos != df_priv_scores).any())

            self.logger.info("Testing predict on single point")

            labels, scores = self.clf.predict(p, return_decision_function=True,
                                              n_jobs=_NoValue)
            self.logger.info("predict(p):\nlabels: {:}\n"
                             "scores: {:}".format(labels, scores))
            _check_classify_scores(labels, scores, 1, self.clf.n_classes)

            # Comparing output of decision_function and predict

            self.assertFalse(
                (df_scores_neg != CArray(scores[:, 0]).ravel()).any())
            self.assertFalse(
                (df_scores_pos != CArray(scores[:, 1]).ravel()).any())

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

        # FIXME: RESTORE AFTER FIXING GRADIENT OF CCLASSSIFIERREJECTDETECTOR (#283)
        # def test_gradient(self):
        #     """Unittest for gradient_f_x method."""
        #     # Training the classifier
        #     clf = self.clf.fit(self.dataset)
        #
        #     import random
        #     pattern = CArray(random.choice(self.dataset.X.get_data()))
        #     self.logger.info("Randomly selected pattern:\n%s", str(pattern))
        #
        #     self._test_gradient_numerical(clf, pattern)
