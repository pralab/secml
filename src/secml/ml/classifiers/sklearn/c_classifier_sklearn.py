"""
.. module:: CClassifierSkLearn
   :synopsis: Generic wrapper for SkLearn classifiers.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from secml.ml.classifiers import CClassifier
from secml.array import CArray
from secml.utils.dict_utils import merge_dicts, SubLevelsDict


class CWrapperSkLearnMixin:
    """Generic wrapper for SkLearn instances.

    Parameters
    ----------
    sklearn_model : `sklearn.base.BaseEstimator` object
        The scikit-learn instance to wrap.

    """

    def __init__(self, sklearn_model):
        self._sklearn_model = sklearn_model

    @property
    def sklearn_model(self):
        """Wrapped SkLearn classifier."""
        return self._sklearn_model

    def get_params(self):
        """Returns the dictionary of class and SkLearn model parameters.

        A parameter is a PUBLIC or READ/WRITE attribute.

        """
        # We extract the PUBLIC (pub) and the READ/WRITE (rw) attributes
        # from the class dictionary, than we build a new dictionary using
        # as keys the attributes names without the accessibility prefix
        # We merge our dict with the sklearn `.get_params()` dict
        return SubLevelsDict(
            merge_dicts(super(CWrapperSkLearnMixin, self).get_params(),
                        self._sklearn_model.get_params()))

    def __getattribute__(self, key):
        """Get an attribute.

        This allow getting also the attributes of the internal sklearn model.

        """
        try:
            # If we are not getting the sklearn model itself
            if key != '_sklearn_model' and hasattr(self, '_sklearn_model'):
                return self._sklearn_model.get_params()[key]
        except KeyError:
            pass  # Parameter not found in sklearn model
        # Try to get the parameter from self
        return super(CWrapperSkLearnMixin, self).__getattribute__(key)

    def __setattr__(self, key, value):
        """Set an attribute.

        This allow setting also the attributes of the internal sklearn model.

        """
        if hasattr(self, '_sklearn_model') and \
                key in self._sklearn_model.get_params():
            self._sklearn_model.set_params(**{key: value})
        else:  # Otherwise, normal python set behavior
            super(CWrapperSkLearnMixin, self).__setattr__(key, value)


class CClassifierSkLearn(CWrapperSkLearnMixin, CClassifier):
    """Generic wrapper for SkLearn classifiers.

    Parameters
    ----------
    sklearn_model : `sklearn.base.BaseEstimator` object
        The scikit-learn model to wrap. Must implement `fit` and either
        `decision_function` or `predict_proba` methods.
    preprocess : CModule or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CModule subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'sklearn-clf'

    """

    __class_type = 'sklearn-clf'

    def __init__(self, sklearn_model, preprocess=None):

        CWrapperSkLearnMixin.__init__(self, sklearn_model)
        CClassifier.__init__(self, preprocess=preprocess)

        if hasattr(sklearn_model, 'classes_'):  # Model is pretrained
            self._classes = CArray(sklearn_model.classes_)
            # FIXME: how to obtain this from pretrained models?
            self._n_features = 0

    def _fit(self, x, y=None):
        """Fit sklearn model.

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features).
        y : CArray
            Array of shape (n_samples,) containing the class labels.

        Returns
        -------
        CClassifierSkLearn
            Trained classifier.

        """
        self._sklearn_model.fit(x.get_data(), y.get_data())
        return self

    def _forward(self, x):
        """Implementation of decision function."""

        if hasattr(self._sklearn_model, "decision_function"):
            scores = self._sklearn_model.decision_function(x.get_data())
            probs = False
        elif hasattr(self._sklearn_model, "predict_proba"):
            scores = self._sklearn_model.predict_proba(x.get_data())
            probs = True
        else:
            raise AttributeError(
                "This model has neither decision_function nor predict_proba.")

        scores = CArray(scores)

        # two-class classifiers outputting only scores for class 1
        if len(scores.shape) == 1:  # duplicate column for class 0
            outputs = CArray.zeros(shape=(x.shape[0], self.n_classes))
            outputs[:, 1] = scores.T
            outputs[:, 0] = -scores.T if probs is False else 1 - scores.T
            scores = outputs

        if scores.shape[1] != self.n_classes:  # this happens in one-vs-one
            raise ValueError(
                "Number of columns is not equal to number of classes!")

        scores.atleast_2d()
        return scores

    def _backward(self, w):
        raise NotImplementedError(
            "`_backward` is not implemented for this generic sklearn wrapper.")
