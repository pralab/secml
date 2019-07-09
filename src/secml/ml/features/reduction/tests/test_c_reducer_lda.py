from secml.ml.features.tests import CPreProcessTestCases

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from secml.array import CArray
from secml.ml.features.reduction import CLDA
from secml.figure import CFigure


class TestCLda(CPreProcessTestCases):
    """Unittests for CLDA."""

    def setUp(self):
        # As our test cases are not always linearly independent,
        # LDA will warn about "Variables are collinear".
        # We can ignore the warning in this context
        self.logger.filterwarnings("ignore", "Variables are collinear.")

        super(TestCLda, self).setUp()

    def test_lda(self):
        """Test for LDA. This compares sklearn equivalent to our method."""

        def sklearn_comp(array, y):
            self.logger.info("Original array is:\n{:}".format(array))

            # Sklearn normalizer
            sklearn_lda = LinearDiscriminantAnalysis().fit(
                array.tondarray(), y.tondarray())
            target = CArray(sklearn_lda.transform(array.tondarray()))
            # Our normalizer
            lda = CLDA().fit(array, y)
            result = lda.transform(array)

            self.logger.info("Sklearn result is:\n{:}".format(target))
            self.logger.info("Result is:\n{:}".format(result))

            self.assert_array_almost_equal(result, target)

        # A min of 2 samples is required by LDA so we cannot test single rows
        sklearn_comp(self.array_dense, CArray([0, 1, 0]))
        sklearn_comp(self.array_sparse, CArray([0, 1, 0]))
        sklearn_comp(self.column_dense, CArray([0, 1, 0]))
        sklearn_comp(self.column_sparse, CArray([0, 1, 0]))

    def test_plot(self):
        """Test for LDA. Check LDA Result Graphically.

        Apply Lda to Sklearn Iris Dataset and compare it with
        "Linear Discriminant Analysis bit by bit" by Sebastian Raschka
        http://sebastianraschka.com/Articles/2014_python_lda.html
        into the plot we must see approximatively:
        x axes: from -2 to -1 virginica, from -1 to 0 versicolor, from 1 to 2,3 setosa
        y axes: from -1 to -1 virginica, from -1 to 0.5 versicolor, from -1 to 1 setosa

        """
        from sklearn.datasets import load_iris

        iris_db = load_iris()
        patterns = CArray(iris_db.data)
        labels = CArray(iris_db.target)

        lda = CLDA()
        lda.fit(patterns, labels)
        # store dataset reduced with pca
        red_dts = lda.fit_transform(patterns, labels)

        fig = CFigure(width=10, markersize=8)
        fig.sp.scatter(red_dts[:, 0].ravel(),
                       red_dts[:, 1].ravel(),
                       c=labels)
        fig.show()

    def test_chain(self):
        """Test a chain of preprocessors."""
        x_chain = self._test_chain(
            self.array_dense,
            ['min-max', 'mean-std', 'lda'],
            [{'feature_range': (-5, 5)}, {}, {}],
            y=CArray([1, 0, 1])  # LDA is supervised
        )

        # Expected shape is (3, 1), as lda max n_components is classes - 1
        self.assertEqual((self.array_dense.shape[0], 1), x_chain.shape)

        x_chain = self._test_chain(
            self.array_dense,
            ['mean-std', 'lda', 'min-max'],
            [{}, {}, {}],
            y=CArray([1, 0, 1])  # LDA is supervised
        )

        # Expected shape is (3, 1), as lda max n_components is classes - 1
        self.assertEqual((self.array_dense.shape[0], 1), x_chain.shape)

    # TODO: ADD TEST FOR GRADIENT (WHEN IMPLEMENTED)


if __name__ == '__main__':
    CPreProcessTestCases.main()
