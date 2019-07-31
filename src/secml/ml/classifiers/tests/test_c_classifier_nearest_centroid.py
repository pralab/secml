from secml.ml.classifiers.tests import CClassifierTestCases

from secml.data.loader import CDLRandom, CDLRandomBlobs
from secml.ml.classifiers import CClassifierNearestCentroid
from secml.ml.features.normalization import CNormalizerMinMax
from secml.utils import fm


class TestCClassifierNearestCentroid(CClassifierTestCases):
    """Unit test for CClassifierNearestCentroid."""

    def setUp(self):
        """Test for init and fit methods."""

        self.dataset = CDLRandom(n_features=2, n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1).load()

        self.dataset.X = CNormalizerMinMax().fit_transform(self.dataset.X)

        self.nc = CClassifierNearestCentroid()

    def test_plot(self):
        """ Compare the classifiers graphically"""
        ds = CDLRandomBlobs(n_samples=100, centers=3, n_features=2,
                            random_state=1).load()
        fig = self._test_plot(self.nc, ds, [-10])
        fig.savefig(fm.join(fm.abspath(__file__), 'figs',
                            'test_c_classifier_nearest_centroid.pdf'))

    def test_fun(self):
        """Test for decision_function() and predict() methods."""
        scores_d = self._test_fun(self.nc, self.dataset.todense())
        scores_s = self._test_fun(self.nc, self.dataset.tosparse())

        self.assert_array_almost_equal(scores_d, scores_s)

    def test_preprocess(self):
        """Test classifier with preprocessors inside."""
        ds = CDLRandom().load()

        # All linear transformations
        self._test_preprocess(ds, self.nc,
                              ['min-max', 'mean-std'],
                              [{'feature_range': (-1, 1)}, {}])

        # Mixed linear/nonlinear transformations
        self._test_preprocess(ds, self.nc,
                              ['pca', 'unit-norm'], [{}, {}])


if __name__ == '__main__':
    CClassifierTestCases.main()
