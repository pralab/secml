"""
.. module:: CClassifierLogistic
   :synopsis: Logistic Regression (aka logit, MaxEnt) classifier

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from sklearn.linear_model import LogisticRegression

from secml.array import CArray
from secml.ml.classifiers import CClassifierLinear
from secml.ml.classifiers.loss import CLoss
from secml.ml.classifiers.gradients import \
    CClassifierGradientLogistic


class CClassifierLogistic(CClassifierLinear):
    """Logistic Regression (aka logit, MaxEnt) classifier.

    Parameters
    ----------
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    """
    __class_type = 'logistic'

    def __init__(self, C=1.0, max_iter=100, random_seed=None, preprocess=None):

        CClassifierLinear.__init__(self, preprocess=preprocess)

        self.C = C
        self.max_iter = max_iter
        self.random_seed = random_seed

        self._classifier_loss = CLoss.create('log')

        self._init_w = None
        self._init_b = None

        self._gradients = CClassifierGradientLogistic()

    @property
    def gradients(self):
        return self._gradients

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def random_seed(self):
        return self._random_seed

    @max_iter.setter
    def max_iter(self, value):
        self._max_iter = int(value)

    @random_seed.setter
    def random_seed(self, value):
        self._random_seed = value

    @property
    def C(self):
        """Penalty parameter C of the error term."""
        return self._C

    @C.setter
    def C(self, value):
        """Set the penalty parameter C of the error term.

        Parameters
        ----------
        value : float
            Penalty parameter C of the error term.

        """
        self._C = float(value)

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b

    @w.setter
    def w(self, value):
        self._w = value

    @b.setter
    def b(self, value):
        self._b = value

    def _init_clf(self):
        self._sklearn_clf = LogisticRegression(
            penalty='l2',
            dual=False,
            tol=0.0001,
            C=self._C,
            fit_intercept=True,
            intercept_scaling=1.0,
            class_weight=None,
            solver='liblinear',
            random_state=self._random_seed,
            max_iter=self._max_iter,
            multi_class='ovr',
            verbose=0,
            warm_start=False,
        )

    def _fit(self, ds):
        """
        Train the classifier.

        The weights and bias initialization is saved the first time that the
        training function is runned and kept fixed

        :param ds:
        :return:
        """
        self._init_clf()

        self._sklearn_clf.fit(ds.X.tondarray(), ds.Y.tondarray())

        self._w = CArray(self._sklearn_clf.coef_)
        self._b = CArray(self._sklearn_clf.intercept_)

        return self
