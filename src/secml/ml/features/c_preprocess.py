"""
.. module:: CPreProcess
   :synopsis: Common interface for feature preprocessing algorithms.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta, abstractmethod
from secml.ml import CModule


class CPreProcess(CModule, metaclass=ABCMeta):
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
        # TODO: CModule.__init__ should handle the call to create.
        preprocess = None if preprocess is None \
            else CPreProcess.create(preprocess)
        CModule.__init__(self, preprocess=preprocess)

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

    def fit_transform(self, x, y=None, caching=False):
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
        caching: bool
                 True if preprocessed x should be cached for backward pass

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
        return self.forward(x, caching=caching)

    @abstractmethod
    def _forward(self, x):
        """Apply the transformation algorithm on x.

        Parameters
        ----------
        x : CArray
            Preprocessed array to be transformed.
            Shape of input array depends on the algorithm itself.

        Returns
        -------
        CArray
            Transformed input data.

        """
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
        return self.forward(x, caching=False)

    def _inverse_transform(self, x):
        raise NotImplementedError(
            "inverting this transformation is not supported.")

    def inverse_transform(self, x):
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

        v = self._inverse_transform(x)

        # Revert data using the inner preprocess, if defined
        if self.preprocess is not None:
            return self.preprocess.inverse_transform(v)

        return v

    # Same doc for the protected method
    _inverse_transform.__doc__ = inverse_transform.__doc__
