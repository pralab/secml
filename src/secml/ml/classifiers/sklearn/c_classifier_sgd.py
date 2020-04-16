"""
.. module:: CClassifierSGD
   :synopsis: Stochastic Gradient Descent (SGD) classifier

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from sklearn import linear_model

from secml.array import CArray
from secml.ml.classifiers import CClassifierLinearMixin, CClassifierSkLearn
from secml.ml.classifiers.loss import CLoss
from secml.ml.classifiers.regularizer import CRegularizer
from secml.ml.classifiers.gradients import CClassifierGradientSGDMixin


class CClassifierSGD(CClassifierLinearMixin,
                     CClassifierSkLearn,
                     CClassifierGradientSGDMixin):
    """Stochastic Gradient Descent Classifier.

    Parameters
    ----------
    loss : CLoss
        Loss function to be used during classifier training.
    regularizer : CRegularizer
        Regularizer function to be used during classifier training.
    kernel : None or CKernel subclass, optional

        .. deprecated:: 0.12

        Instance of a CKernel subclass to be used for computing similarity
        between patterns. If None (default), a linear SVM will be created.
        In the future this parameter will be removed from this classifier and
        kernels will have to be passed as preprocess.
    alpha : float, optional
        Constant that multiplies the regularization term. Default 0.01.
        Also used to compute learning_rate when set to 'optimal'.
    fit_intercept : bool, optional
        If True (default), the intercept is calculated, else no intercept will
        be used in calculations (e.g. data is expected to be already centered).
    max_iter : int, optional
        The maximum number of passes over the training data (aka epochs).
        Default 1000.
    tol : float or None, optional
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > best_loss - tol) for 5 consecutive epochs. Default None.
    shuffle : bool, optional
        If True (default) the training data is shuffled after each epoch.
    learning_rate : str, optional
        The learning rate schedule. If 'constant', eta = eta0;
        if 'optimal' (default), eta = 1.0 / (alpha * (t + t0)), where t0 is
        chosen by a heuristic proposed by Leon Bottou; if 'invscaling',
        eta = eta0 / pow(t, power_t); if 'adaptive', eta = eta0, as long as
        the training keeps decreasing.
    eta0 : float, optional
        The initial learning rate for the 'constant', 'invscaling' or
        'adaptive' schedules. Default 10.0.
    power_t : float, optional
        The exponent for inverse scaling learning rate. Default 0.5.
    class_weight : {dict, 'balanced', None}, optional
        Set the parameter C of class i to `class_weight[i] * C`.
        If not given (default), all classes are supposed to have
        weight one. The 'balanced' mode uses the values of labels to
        automatically adjust weights inversely proportional to
        class frequencies as `n_samples / (n_classes * np.bincount(y))`.
    warm_start : bool, optional
        If True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Default False.
    average : bool or int, optional
        If True, computes the averaged SGD weights and stores the result in
        the `coef_` attribute. If set to an int greater than 1, averaging
        will begin once the total number of samples seen reaches average.
        Default False.
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
    class_type : 'sgd'

    """
    __class_type = 'sgd'

    def __init__(self, loss, regularizer, alpha=0.01,
                 fit_intercept=True, max_iter=1000, tol=None,
                 shuffle=True, learning_rate='optimal',
                 eta0=10.0, power_t=0.5, class_weight=None,
                 warm_start=False, average=False, random_state=None,
                 preprocess=None):

        # Keep private (not an sklearn sgd parameter)
        self._loss = CLoss.create(loss)
        # Keep private (not an sklearn sgd parameter)
        self._regularizer = CRegularizer.create(regularizer)

        sklearn_model = linear_model.SGDClassifier(
            loss=self.loss.class_type,
            penalty=self.regularizer.class_type,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            class_weight=class_weight,
            average=average,
            warm_start=warm_start,
            random_state=random_state)

        # Pass loss function parameters to classifier
        sklearn_model.set_params(**self.loss.get_params())
        # Pass regularizer function parameters to classifier
        sklearn_model.set_params(**self.regularizer.get_params())

        # Calling the superclass init
        CClassifierSkLearn.__init__(self, sklearn_model=sklearn_model,
                                    preprocess=preprocess)

    @property
    def loss(self):
        """Returns the loss function used by classifier."""
        return self._loss

    @property
    def regularizer(self):
        """Returns the regularizer function used by classifier."""
        return self._regularizer

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
