from secml.testing import CUnitTest

from secml.array import CArray
from secml.ml.peval.metrics import CRoc
from secml.figure import CFigure
from secml.ml.classifiers import CClassifierSVM
from secml.data.loader import CDLRandom


class TestCRoc(CUnitTest):
    """Unit test for CPlotMetric (ROC plots)."""

    def setUp(self):

        self.ds_loader = CDLRandom(n_features=1000, n_redundant=200,
                                   n_informative=250, n_clusters_per_class=2)
        self.ds1 = self.ds_loader.load()
        self.ds2 = self.ds_loader.load()

        self.y1 = self.ds1.Y
        self.y2 = self.ds2.Y

        self.svm = CClassifierSVM(C=1e-7).fit(self.ds1)

        _, self.s1 = self.svm.predict(
            self.ds1.X, return_decision_function=True)
        _, self.s2 = self.svm.predict(
            self.ds2.X, return_decision_function=True)

        self.s1 = self.s1[:, 1].ravel()
        self.s2 = self.s2[:, 1].ravel()

        # Roc with not computed average (2 repetitions)
        self.roc_nomean = CRoc()
        self.roc_nomean.compute([self.y1, self.y2], [self.s1, self.s2])

        # Roc with average (2 repetitions)
        self.roc_wmean = CRoc()
        self.roc_wmean.compute([self.y1, self.y2], [self.s1, self.s2])
        self.roc_wmean.average()

    def test_standard(self):
        """Plot of standard ROC."""

        # Testing without input CFigure
        roc_plot = CFigure()
        roc_plot.sp.title('ROC Curve Standard')
        # Plotting 2 times (to show multiple curves)
        # add one curve for repetition and call it rep 0 and rep 1 of roc 1
        roc_plot.sp.plot_roc(self.roc_wmean.mean_fpr, self.roc_wmean.mean_tpr)

        roc_plot.show()

    def test_mean(self):
        """Plot of average ROC."""

        # Testing without input CFigure
        roc_plot = CFigure()
        roc_plot.sp.title('ROC Curve')
        # Plotting 2 times (to show 2 curves)
        roc_plot.sp.plot_roc_mean(self.roc_wmean, label='roc1 mean', plot_std=True)
        roc_plot.sp.plot_roc_reps(self.roc_wmean, label='roc1')

        roc_plot.show()

        # Testing mean plot with no average
        with self.assertRaises(ValueError):
            roc_plot.sp.plot_roc_mean(self.roc_nomean)

    def test_custom_params(self):
        """Plot of ROC altering default parameters."""

        # Testing without input CFigure
        roc_plot = CFigure()
        roc_plot.sp.title('ROC Curve - Custom')
        roc_plot.sp.xlim(0.1, 100)
        roc_plot.sp.ylim(30, 100)
        roc_plot.sp.yticks([70, 80, 90, 100])
        roc_plot.sp.yticklabels(['70', '80', '90', '100'])
        # Plotting 2 times (to show 2 curves)
        roc_plot.sp.plot_roc_mean(self.roc_wmean, label='roc1')
        roc_plot.sp.plot_roc_mean(self.roc_wmean, label='roc2')

        roc_plot.show()

    def test_single(self):
        """Plot of ROC repetitions."""

        # Testing without input CFigure
        roc_plot = CFigure()
        roc_plot.sp.title('ROC Curve Repetitions')
        # Plotting 2 times (to show multiple curves)
        # add one curve for repetition and call it rep 0 and rep 1 of roc 1
        roc_plot.sp.plot_roc_reps(self.roc_nomean, label='roc1')
        # add one curve for repetition and call it rep 0 and rep 1 of roc 2
        roc_plot.sp.plot_roc_reps(self.roc_nomean, label='roc2')

        roc_plot.show()

    def test_compare_sklearn(self):

        import numpy as np

        from sklearn import svm, datasets
        from sklearn.metrics import roc_curve, auc
        from sklearn.model_selection import StratifiedKFold

        from secml.figure import CFigure
        roc_fig = CFigure(width=12)

        # import some data to play with
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        X, y = X[y != 2], y[y != 2]
        n_samples, n_features = X.shape

        # Add noisy features
        random_state = np.random.RandomState(0)
        X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

        # Classification and ROC analysis

        # Run classifier with cross-validation and plot ROC curves
        classifier = svm.SVC(kernel='linear', probability=True,
                             random_state=random_state)

        roc_fig.subplot(1, 2, 1)

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 1000)

        cv = StratifiedKFold(n_splits=6)
        for i, (train, test) in enumerate(cv.split(X, y)):
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            roc_fig.sp.plot(fpr, tpr, linewidth=1,
                            label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        roc_fig.sp.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6),
                        label='Luck')

        mean_tpr /= cv.get_n_splits()
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        roc_fig.sp.plot(mean_fpr, mean_tpr, 'k--',
                        label='Mean ROC (area = %0.2f)' % mean_auc,
                        linewidth=2)

        roc_fig.sp.xlim([-0.05, 1.05])
        roc_fig.sp.ylim([-0.05, 1.05])
        roc_fig.sp.xlabel('False Positive Rate')
        roc_fig.sp.ylabel('True Positive Rate')
        roc_fig.sp.title('Sklearn Receiver operating characteristic example')
        roc_fig.sp.legend(loc="lower right")
        roc_fig.sp.grid()

        self.logger.info("Plotting using our CPLotRoc")

        roc_fig.subplot(1, 2, 2)

        score = []
        true_y = []
        for i, (train, test) in enumerate(cv.split(X, y)):
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            true_y.append(CArray(y[test]))
            score.append(CArray(probas_[:, 1]))

        self.roc_wmean = CRoc()
        self.roc_wmean.compute(true_y, score)
        fp, tp = self.roc_wmean.average()

        roc_fig.sp.plot([0, 100], [0, 100], '--', color=(0.6, 0.6, 0.6),
                        label='Luck')

        roc_fig.sp.xticks([0, 20, 40, 60, 80, 100])
        roc_fig.sp.xticklabels(['0', '20', '40', '60', '80', '100'])

        roc_fig.sp.plot_roc_mean(
            self.roc_wmean, plot_std=True, logx=False, style='go-',
            label='Mean ROC (area = %0.2f)' % (auc(fp.tondarray(),
                                                   tp.tondarray())))

        roc_fig.sp.xlim([-0.05 * 100, 1.05 * 100])
        roc_fig.sp.ylim([-0.05 * 100, 1.05 * 100])
        roc_fig.sp.title('SecML Receiver operating characteristic example')
        roc_fig.sp.legend(loc="lower right")
        roc_fig.show()


if __name__ == '__main__':
    CUnitTest.main()
