from secml.utils import CUnitTest

from secml.explanation import CExplainerLocalLinear
from secml.ml.classifiers import CClassifierSVM
from secml.data.loader import CDLRandomToy
from secml.array import CArray
from secml.figure import CFigure


class TestCExplainerLocalLinear(CUnitTest):
    """Unittests for CExplainerLocalLinear"""

    def setUp(self):

        self.clf = CClassifierSVM()
        # 100 samples, 2 classes, 20 features
        self.ds = CDLRandomToy('digits', class_list=[0, 1], zero_one=True).load()

        # Training classifier
        self.clf.fit(self.ds)

        self.explainer = CExplainerLocalLinear(self.clf)

    def test_explain(self):
        """Unittest for explain method."""
        i = 67
        x = self.ds.X[i, :]

        attr = self.explainer.explain(x)

        self.logger.info("Attributions:\n{:}".format(attr.tolist()))

        self.assertIsInstance(attr, CArray)
        self.assertEqual(attr.shape, attr.shape)

        fig = CFigure(height=3, width=6)

        # Plotting original image
        fig.subplot(1, 2, 1)
        fig.sp.imshow(attr.reshape((8, 8)), cmap='gray')

        th = max(abs(attr.min()), abs(attr.max()))

        # Plotting attributions
        fig.subplot(1, 2, 2)
        fig.sp.imshow(attr.reshape((8, 8)),
                      cmap='seismic', vmin=-1*th, vmax=th)

        fig.show()


if __name__ == '__main__':
    CUnitTest.main()
