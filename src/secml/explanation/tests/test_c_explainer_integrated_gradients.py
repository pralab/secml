from secml.testing import CUnitTest

from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF
from secml.data.loader import CDLDigits
from secml.array import CArray
from secml.figure import CFigure

from secml.explanation import CExplainerIntegratedGradients


class TestCExplainerIntegratedGradients(CUnitTest):
    """Unittests for CExplainerIntegratedGradients"""

    @classmethod
    def setUpClass(cls):

        CUnitTest.setUpClass()

        # 100 samples, 2 classes, 20 features
        cls.ds = CDLDigits().load()

        cls.clf = CClassifierMulticlassOVA(
            CClassifierSVM, kernel=CKernelRBF(gamma=1e-3))

        # Training classifier
        cls.clf.fit(cls.ds)

    def setUp(self):

        self.explainer = CExplainerIntegratedGradients(self.clf)
        self.explainer.verbose = 2

    def test_explain(self):
        """Unittest for explain method."""
        i = 67
        ds_i = self.ds[i, :]
        x, y_true = ds_i.X, ds_i.Y.item()

        self.logger.info("Explaining P{:} c{:}".format(i, y_true))

        x_pred, x_score = self.clf.predict(x, return_decision_function=True)

        self.logger.info(
            "Predicted class {:}, scores:\n{:}".format(x_pred.item(), x_score))
        self.logger.info("Candidates: {:}".format(x_score.argsort()[::-1]))

        ref_img = None  # Use default reference image
        m = 100  # Number of steps

        attr = CArray.empty(shape=(0, x.shape[1]), sparse=x.issparse)
        for c in self.ds.classes:  # Compute attributions for each class
            a = self.explainer.explain(x, y=c, reference=ref_img, m=m)
            attr = attr.append(a, axis=0)

        self.assertIsInstance(attr, CArray)
        self.assertEqual(attr.shape[1], x.shape[1])
        self.assertEqual(attr.shape[0], self.ds.num_classes)

        fig = CFigure(height=1.5, width=12)

        # Plotting original image
        fig.subplot(1, self.ds.num_classes+1, 1)
        fig.sp.imshow(x.reshape((8, 8)), cmap='gray')
        fig.sp.title("Origin c{:}".format(y_true))
        fig.sp.yticks([])
        fig.sp.xticks([])

        th = max(abs(attr.min()), abs(attr.max()))

        # Plotting attributions
        for c in self.ds.classes:
            fig.subplot(1, self.ds.num_classes+1, 2+c)
            fig.sp.imshow(attr[c, :].reshape((8, 8)),
                          cmap='seismic', vmin=-1*th, vmax=th)
            fig.sp.title("Attr c{:}".format(c))
            fig.sp.yticks([])
            fig.sp.xticks([])

        fig.tight_layout()
        fig.show()

    def test_linear_interpolation(self):
        """Unittest for linear interpolation method."""
        i = 10
        sample = self.ds.X[i, :]

        ref_img = None  # Use default reference image
        m = 50  # Number of steps

        ret = self.explainer.linearly_interpolate(sample, ref_img, m)

        self.assertEqual(m, len(ret))
        self.assertIsInstance(ret[10], CArray)
        self.assertEqual(ret[10].shape, sample.shape)


if __name__ == '__main__':
    CUnitTest.main()
