"""
.. module:: CPreProcess
   :synopsis: Common interface for feature preprocessing algorithms.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta, abstractmethod
import six

from secml.core import CCreator


@six.add_metaclass(ABCMeta)
class CPreProcess(CCreator):
    """Common interface for feature preprocessing algorithms.

    Parameters
    ----------
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    """
    __super__ = 'CPreProcess'

    def __init__(self, preprocess=None):
        self._preprocess = None if preprocess is None \
            else CPreProcess.create(preprocess)

    @property
    def preprocess(self):
        """Inner preprocessor (if any)."""
        return self._preprocess

    @staticmethod
    def create_chain(class_items, kwargs_list):
        """Creates a chain of preprocessors.

        Parameters
        ----------
        class_items : list of str or class instances
            A list of mixed class types or CPreProcess instances.
            The object created with the first type/instance of the list
            will be the preprocess of the object created using the second
            type/instance in the list and so on until the end of the list.
        kwargs_list : list of dict
            A list of dictionaries, one for each item in `class_items`,
            to specify any additional argument for each specific preprocessor.

        Returns
        -------
        CPreProcess
            The chain of preprocessors.

        """
        chain = None
        for i, pre_id in enumerate(class_items):
            chain = CPreProcess.create(
                pre_id, preprocess=chain, **kwargs_list[i])

        return chain

    @abstractmethod
    def _check_is_fitted(self):
        """Check if the preprocessor is trained (fitted).

        Raises
        ------
        NotFittedError
            If the preprocessor is not fitted.

        """
        raise NotImplementedError

    def _preprocess_data(self, x):
        """Apply the inner preprocess to input, if defined.

        Parameters
        ----------
        x : CArray
            Data to be transformed using inner preprocess, if defined.

        Returns
        -------
        CArray
            If an inner preprocess is defined, will be the transformed data.
            Otherwise input data is returned as is.

        """
        if self.preprocess is not None:
            return self.preprocess.transform(x)
        return x

    @abstractmethod
    def _fit(self, x, y=None):
        raise NotImplementedError("training of preprocessor not implemented.")

    def fit(self, x, y=None):
        """Fit transformation algorithm.

        Parameters
        ----------
        x : CArray
            Array to be used for training.
            Shape of input array depends on the algorithm itself.
        y : CArray or None, optional
            Flat array with the label of each pattern.
            Can be None if not required by the preprocessing algorithm.

        Returns
        -------
        CPreProcess
            Instance of the trained preprocessor.

        """
        if self.preprocess is not None:
            x = self.preprocess.fit_transform(x, y)

        return self._fit(x, y)

    _fit.__doc__ = fit.__doc__  # Same doc for the protected method

    def fit_transform(self, x, y=None):
        """Fit preprocessor using data and then transform data.

        This method is equivalent to call fit(data) and transform(data)
        in sequence, but it's useful when data is both the training array
        and the array to be transformed.

        Parameters
        ----------
        x : CArray
            Array to be transformed.
            Each row must correspond to one single patterns, so each
            column is a different feature.
        y : CArray or None, optional
            Flat array with the label of each pattern.
            Can be None if not required by the preprocessing algorithm.

        Returns
        -------
        CArray
            Transformed input data.

        See Also
        --------
        fit : fit the preprocessor.
        transform : transform input data.

        """
        self.fit(x, y)  # train preprocessor first
        return self.transform(x)

    @abstractmethod
    def _transform(self, x):
        raise NotImplementedError("`transform` not implemented.")

    def transform(self, x):
        """Apply the transformation algorithm on data.

        Parameters
        ----------
        x : CArray
            Array to be transformed.
            Shape of input array depends on the algorithm itself.

        Returns
        -------
        CArray
            Transformed input data.

        """
        self._check_is_fitted()

        # Transform data using inner preprocess, if defined
        x = self._preprocess_data(x)

        return self._transform(x)

    _transform.__doc__ = transform.__doc__  # Same doc for the protected method

    def _revert(self, x):
        raise NotImplementedError(
            "reverting this transformation is not supported.")

    def revert(self, x):
        """Revert data to original form.

        Parameters
        ----------
        x : CArray
            Transformed array to be reverted to original form.
            Shape of input array depends on the algorithm itself.

        Returns
        -------
        CArray
            Original input data.

        Warnings
        --------
        Reverting a transformed array is not always possible.
        See description of each preprocessor for details.

        """
        self._check_is_fitted()

        v = self._revert(x)

        # Revert data using the inner preprocess, if defined
        if self.preprocess is not None:
            return self.preprocess.revert(v)

        return v

    _revert.__doc__ = revert.__doc__  # Same doc for the protected method

    def _gradient(self, x, w=None):
        raise NotImplementedError("gradient is not implemented for {:}"
                                  "".format(self.__class__.__name__))

    def gradient(self, x, w=None):
        """Returns the preprocessor gradient wrt data.

        Parameters
        ----------
        x : CArray
            Data array, 2-Dimensional or ravel.
        w : CArray or None, optional
            If CArray, will be left-multiplied to the gradient
            of the preprocessor.

        Returns
        -------
        gradient : CArray
            Gradient of the preprocessor wrt input data.
            Array of shape (x.shape[1], x.shape[1]) if `w` is None,
            otherwise an array of shape (w.shape[0], x.shape[1]).
            If `w.shape[0]` is 1, result will be raveled.

        """
        self._check_is_fitted()

        x_in = x  # Original input data (not transformed by inner preprocess)

        # Input should be transformed using the inner preprocessor, if defined
        x = self._preprocess_data(x)

        grad = self._gradient(x, w=w)

        if self.preprocess is not None:  # Use original input data
            grad = self.preprocess.gradient(x_in, w=grad)

        return grad.ravel() if grad.is_vector_like else grad

    _gradient.__doc__ = gradient.__doc__  # Same doc for the protected method
