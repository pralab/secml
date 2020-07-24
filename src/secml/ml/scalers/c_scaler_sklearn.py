"""
.. module:: CScalerSkLearn
   :synopsis: Generic wrapper for SkLearn scalers.

.. moduleauthor:: Marco Meloni <m.meloni42@studenti.unica.it>

"""
from secml.ml import CModule
from secml.array import CArray
from secml.ml.classifiers.sklearn.c_classifier_sklearn \
    import CWrapperSkLearnMixin
from abc import ABCMeta, abstractmethod


class CScalerSkLearn(CWrapperSkLearnMixin, CModule, metaclass=ABCMeta):
    """Generic wrapper for SkLearn scalers.

    Parameters
    ----------
    sklearn_scaler : `sklearn.preprocessing` object
        The scikit-learn scaler to wrap. Must implement `fit`, `transform`
        and `fit_transform`.
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    """
    __super__ = 'CScalerSkLearn'

    def __init__(self, sklearn_scaler, preprocess=None):

        CModule.__init__(self, preprocess=preprocess)
        self._sklearn_scaler = sklearn_scaler

    @property
    def sklearn_scaler(self):
        """Wrapped SkLearn classifier."""
        return self._sklearn_scaler

    def _fit(self, x, y=None):
        """Compute parameters for later scaling.

        Parameters
        ----------
        x : `CArray` object
            The data used to compute the parameters for later scaling
        y :
            Ignored

        """
        self._sklearn_scaler.fit(x.get_data(), y)
        return self

    @abstractmethod
    def _check_is_fitted(self):
        pass

    @staticmethod
    def _check_is_fitted_scaler(scaler, attributes, msg=None, check_all=True):
        """Check if the input object is trained (fitted).

        Checks if the input object is fitted by verifying if all or any of the
        input attributes are not None.

        Parameters
        ----------
        scaler : object
            Instance of the class to check. Must implement `.fit()` method.
        attributes : str or list of str
            Attribute or list of attributes to check.
            Es.: `['classes', 'n_features', ...], 'classes'`
        msg : str or None, optional
            If None, the default error message is:
            "this `{name}` is not trained. Call `.fit()` first.".
            For custom messages if '{name}' is present in the message string,
            it is substituted by the class name of the checked object.
        check_all : bool, optional
            Specify whether to check (True) if all of the given attributes
            are not None or (False) just any of them. Default True.

        Raises
        ------
        NotFittedError
            If `check_all` is True and any of the attributes is None;
            if `check_all` is False and all of attributes are None.

        """
        from secml.core.type_utils import is_list, is_str
        from secml.core.exceptions import NotFittedError

        if msg is None:
            msg = "this `{name}` is not trained. Call `._fit()` first."

        if is_str(attributes):
            attributes = [attributes]
        elif not is_list(attributes):
            raise TypeError(
                "the attribute(s) to check must be a string or a list "
                "of strings")

        obj = scaler.sklearn_scaler

        condition = any if check_all is True else all

        if condition([hasattr(obj, attr) is False for attr in attributes]):
            raise NotFittedError(msg.format(name=scaler.__class__.__name__))

    def _forward(self, x):
        return CArray(self._sklearn_scaler.transform(x.get_data()))

    def _backward(self, w):
        raise NotImplementedError()

