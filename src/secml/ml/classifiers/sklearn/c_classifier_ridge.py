"""
.. module:: CClassifierRidge
   :synopsis: Ridge classifier

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from sklearn.linear_model import RidgeClassifier

from secml.array import CArray
from secml.ml.classifiers import CClassifierLinearMixin, CClassifierSkLearn
from secml.ml.classifiers.gradients import CClassifierGradientRidgeMixin
from secml.ml.classifiers.loss import CLossSquare
from secml.ml.classifiers.regularizer import CRegularizerL2


class CClassifierRidge(CClassifierLinearMixin, CClassifierSkLearn,
                       CClassifierGradientRidgeMixin):
    """Ridge Classifier.

    Parameters
    ----------
    alpha : float, optional
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Default 1.0.
    max_iter : int, optional
        Maximum number of iterations for conjugate gradient solver.
        Default 1e5.
    class_weight : {dict, 'balanced', None}, optional
        Set the parameter C of class i to `class_weight[i] * C`.
        If not given (default), all classes are supposed to have
        weight one. The 'balanced' mode uses the values of labels to
        automatically adjust weights inversely proportional to
        class frequencies as `n_samples / (n_classes * np.bincount(y))`.
    tol : float, optional
        Precision of the solution. Default 1e-4.
    fit_intercept : bool, optional
        If True (default), the intercept is calculated, else no intercept will
        be used in calculations (e.g. data is expected to be already centered).
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'ridge'

    """
    __class_type = 'ridge'

    _loss = CLossSquare()
    _reg = CRegularizerL2()

    def __init__(self, alpha=1.0, max_iter=1e5, class_weight=None, tol=1e-4,
                 fit_intercept=True, preprocess=None):
        # create instance of sklearn model
        sklearn_model = RidgeClassifier(alpha=alpha,
                                        fit_intercept=fit_intercept,
                                        tol=tol,
                                        max_iter=max_iter,
                                        class_weight=class_weight,
                                        solver='auto')

        # Calling the superclass init
        CClassifierSkLearn.__init__(self, sklearn_model=sklearn_model,
                                    preprocess=preprocess)

    @property
    def C(self):
        """Constant that multiplies the regularization term.

        Equal to 1 / alpha.

        """
        return 1.0 / self.alpha

    @property
    def w(self):
        if self.is_fitted():
            return CArray(self._sklearn_model.coef_).ravel()
        else:
            return None

    @property
    def b(self):
        if self.is_fitted():
            return CArray(self._sklearn_model.intercept_[0])[0] if \
                self.fit_intercept else 0
        else:
            return None

    def _check_input(self, x, y=None):
        """Check if y contains only two classes."""
        x, y = CClassifierSkLearn._check_input(self, x, y)
        if y is not None and y.unique().size != 2:
            raise ValueError("The data (x,y) has more than two classes.")
        return x, y
