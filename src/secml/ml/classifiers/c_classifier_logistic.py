"""
.. module:: CClassifierLogistic
   :synopsis: Logistic Regression (aka logit, MaxEnt) classifier

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from sklearn.linear_model import LogisticRegression

from secml.array import CArray
from secml.ml.classifiers import CClassifierLinear
from secml.ml.classifiers.loss import CLossLogistic
from secml.ml.classifiers.regularizer import CRegularizerL2

from secml.ml.classifiers.gradients import \
    CClassifierGradientLogisticMixin


class CClassifierLogistic(CClassifierLinear, CClassifierGradientLogisticMixin):
    """Logistic Regression (aka logit, MaxEnt) classifier.

    Parameters
    ----------
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    """
    __class_type = 'logistic'

    _loss = CLossLogistic()
    _reg = CRegularizerL2()

    def __init__(self, C=1.0, max_iter=100, random_seed=None, preprocess=None):

        CClassifierLinear.__init__(self, preprocess=preprocess)

        self.C = C
        self.max_iter = max_iter
        self.random_seed = random_seed

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

    def _fit(self, dataset):
        """Trains the One-Vs-All Logistic classifier.

        The following is a private method computing one single
        binary (2-classes) classifier of the OVA schema.

        Representation of each classifier attribute for the multiclass
        case is explained in corresponding property description.

        Parameters
        ----------
        dataset : CDataset
            Binary (2-classes) training set. Must be a :class:`.CDataset`
            instance with patterns data and corresponding labels.

        Returns
        -------
        trained_cls : classifier
            Instance of the used solver trained using input dataset.

        """
        self._init_clf()

        self._sklearn_clf.fit(dataset.X.get_data(), dataset.Y.tondarray())

        self._w = CArray(
            self._sklearn_clf.coef_, tosparse=dataset.issparse).ravel()
        self._b = CArray(self._sklearn_clf.intercept_[0])[0]

        return self
