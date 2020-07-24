"""
.. module:: CClassifierLogistic
   :synopsis: Logistic Regression (aka logit, MaxEnt) classifier

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from sklearn.linear_model import LogisticRegression

from secml.array import CArray
from secml.ml.classifiers import CClassifierLinearMixin, CClassifierSkLearn
from secml.ml.classifiers.loss import CLossLogistic
from secml.ml.classifiers.regularizer import CRegularizerL2

from secml.ml.classifiers.gradients import \
    CClassifierGradientLogisticMixin


class CClassifierLogistic(CClassifierLinearMixin,
                          CClassifierSkLearn,
                          CClassifierGradientLogisticMixin):
    """Logistic Regression (aka logit, MaxEnt) classifier.

    Parameters
    ----------
    C : float, optional
        Penalty parameter C of the error term. Default 1.0.
    max_iter : int, optional
        Maximum number of iterations taken for the solvers to converge.
        Default 100.
    random_state : int, RandomState or None, optional
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Default None.
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'logistic'

    """
    __class_type = 'logistic'

    _loss = CLossLogistic()
    _reg = CRegularizerL2()

    def __init__(self, C=1.0, max_iter=100,
                 random_state=None, preprocess=None):
        sklearn_model = LogisticRegression(
            penalty='l2',
            dual=False,
            tol=0.0001,
            C=C,
            fit_intercept=True,
            intercept_scaling=1.0,
            class_weight=None,
            solver='liblinear',
            random_state=random_state,
            max_iter=max_iter,
            multi_class='ovr',
            verbose=0,
            warm_start=False)

        CClassifierSkLearn.__init__(self, sklearn_model, preprocess=preprocess)

    @property
    def w(self):
        if self.is_fitted():
            return CArray(self._sklearn_model.coef_).ravel()
        else:
            return None

    @property
    def b(self):
        if self.is_fitted():
            return CArray(self._sklearn_model.intercept_[0])[0]
        else:
            return None

    def _check_input(self, x, y=None):
        """Check if y contains only two classes."""
        x, y = CClassifierSkLearn._check_input(self, x, y)
        if y is not None and y.unique().size != 2:
            raise ValueError("The data (x,y) has more than two classes.")
        return x, y
