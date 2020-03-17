from secml.testing import CUnitTest

from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.kernels import CKernelRBF
from secml.data.loader import CDLDigits
from secml.array import CArray
from secml.figure import CFigure

from secml.explanation import CExplainerGradientInput


class TestCExplainerGradientInput(CUnitTest):
    """Unittests for CExplainerGradientInput"""

    def setUp(self):

        # 100 samples, 2 classes, 20 features
        self.ds = CDLDigits().load()

        self.clf = CClassifierMulticlassOVA(
            CClassifierSVM, kernel=CKernelRBF(gamma=1e-3))

        # Training classifier
        self.clf.fit(self.ds)

        self.explainer = CExplainerGradientInput(self.clf)

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

        fig = CFigure(height=1.5, width=12)

        # Plotting original image
        fig.subplot(1, self.ds.num_classes+1, 1)
        fig.sp.imshow(x.reshape((8, 8)), cmap='gray')
        fig.sp.title("Origin c{:}".format(y_true))
        fig.sp.yticks([])
        fig.sp.xticks([])

        attr = CArray.empty(shape=(self.ds.num_classes, x.size))

        # Computing attributions
        for c in self.ds.classes:

            attr_c = self.explainer.explain(x, y=c)
            attr[c, :] = attr_c
            self.logger.info(
                "Attributions class {:}:\n{:}".format(c, attr_c.tolist()))

            self.assertIsInstance(attr, CArray)
            self.assertEqual(attr.shape, attr.shape)

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


if __name__ == '__main__':
    CUnitTest.main()
