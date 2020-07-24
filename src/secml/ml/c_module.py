"""
.. module:: CModule
   :synopsis: Common interface for implementing pre-processing chains and
    automatic differentiation with forward/backward passes.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Angelo Sotgiu <angelo.sotgiu@unica.it>

"""
from abc import ABCMeta, abstractmethod
from secml.core import CCreator
from secml.array import CArray


class CModule(CCreator, metaclass=ABCMeta):
    """Common interface for handling pre-processing chains and implementing
     automatic differentiation with forward/backward passes.

    Parameters
    ----------
    preprocess : CModule or None, optional
        Feature preprocessing to be applied to input data.
        Can be a CModule subclass. If None, input data is used as is.

    n_jobs : int, optional
        Number of parallel workers to use for training the classifier.
        Cannot be higher than processor's number of cores. Default is 1.

    """
    __super__ = 'CModule'

    def __init__(self, preprocess=None, n_jobs=1):
        self._cached_x = None  # cached internal x repr. for backward pass
        self.preprocess = preprocess  # call setter
        self.n_jobs = n_jobs

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
            chain = CModule.create(
                pre_id, preprocess=chain, **kwargs_list[i])

        return chain

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = int(value)

    @property
    def _grad_requires_forward(self):
        """Returns True if gradient requires calling forward besides just
        computing the pre-processed input x. This is useful for modules that
        use auto-differentiation, like PyTorch, or if caching is required
        during the forward step (e.g., in exponential kernels).
        It is False by default for modules in this library, as we compute
        gradients analytically and only require the pre-processed input x."""
        return False

    @abstractmethod
    def _check_is_fitted(self):
        """Checks if the module is trained (fitted).

        Raises
        ------
        NotFittedError
            If the module is not fitted.

        """
        raise NotImplementedError

    def _check_input(self, x, y=None):
        """Checks if input x and y can be casted to CArray, respectively
        as a matrix of shape=(n_samples, n_features) and
        as a vector of shape=(n_samples,)

        Parameters
        ----------
        x : CArray (or compatible)
            Matrix of input samples with shape=(n_samples, n_features)

        y: CArray (or compatible) or None
            Class labels with shape=(n_samples,)

        Raises
        ------
        TypeError
            if x or y are not properly formatted or cannot be casted to the
            desired format.

        Returns
        -------
        x: CArray
            Matrix of input samples with shape=(n_samples, n_features)
        y: CArray or None
            Class labels with shape=(n_samples,) or None (if y is not passed).

        """
        x = CArray(x).atleast_2d()  # Ensuring input is 2-D
        if y is not None:
            y = CArray(y).ravel()
        return x, y

    def _clear_cache(self):
        """Clears cached values within this class instance."""
        self._cached_x = None

    @property
    def preprocess(self):
        """Inner preprocessor (if any)."""
        return self._preprocess

    @preprocess.setter
    def preprocess(self, preprocess):
        self._preprocess = None if preprocess is None \
            else CModule.create(preprocess)

    def _forward_preprocess(self, x, caching=True):
        """Runs forward through the pre-processing chain,
        preparing the input for `_forward` and `_backward`.

        It checks if x has the proper format and the current module is fitted.
        Then, it applies inner pre-processing (if defined), and caches
        the input data for backward pass (when required).

        Parameters
        ----------
        x : CArray
            Input data to be transformed via pre-processing, if set.

        caching: bool
            True if preprocessed input should be cached for backward pass.

        Returns
        -------
        CArray
            Either the input data x (if pre-processing is not set), or its
            transformed version after pre-processing.

        """
        if self.preprocess is not None:
            # apply pre-processing to x
            x_prc = self.preprocess.forward(x)
        else:
            # use directly x as input to this module
            x_prc = x

        if caching is True:
            # cache intermediate representation of x if required,
            # e.g., if backward has to be called after forward.
            self._cached_x = x_prc

        return x_prc

    def forward(self, x, caching=True):
        """Forward pass on input x.
        This function internally calls self._preprocess_data(x) to handle
        caching of intermediate representation of the input data x.

        Parameters
        ----------
        x : CArray
            Input array to be transformed.
            Shape of input array depends on the algorithm itself.

        caching: bool
            True if preprocessed input should be cached for backward pass.

        Returns
        -------
        CArray
            Transformed input data.

        """
        x, y = self._check_input(x)
        self._check_is_fitted()
        self._clear_cache()

        # Transform data using inner preprocess, if defined
        x = self._forward_preprocess(x=x, caching=caching)
        return self._forward(x)

    @abstractmethod
    def _forward(self, x):
        """Forward pass on input x.

        Parameters
        ----------
        x : CArray
            preprocessed array, ready to be transformed by the current module.

        Returns
        -------
        CArray
            Transformed input data.

        """
        raise NotImplementedError("`_forward` not implemented.")

    def backward(self, w=None):
        """Returns the preprocessor gradient wrt data.

        Parameters
        ----------
        w : CArray or None
            if CArray, it is pre-multiplied to the gradient
            of the module, as in standard reverse-mode autodiff.

        Returns
        -------
        gradient : CArray
            Accumulated gradient of the module wrt input data.

        """

        if self._cached_x is None:
            raise ValueError("Please run forward with caching=True first.")

        grad = self._backward(w=w)

        if self.preprocess is not None:  # accumulate gradients
            grad = self.preprocess.backward(w=grad)

        return grad.ravel() if grad.is_vector_like else grad

    def _backward(self, w):
        raise NotImplementedError("`_backward` is not implemented for {:}"
                                  "".format(self.__class__.__name__))

    _backward.__doc__ = backward.__doc__  # Same doc for the protected method

    @abstractmethod
    def _fit(self, x, y):
        raise NotImplementedError("Fit is not implemented.")

    def fit(self, x, y):
        """Fit estimator.

        Parameters
        ----------
        x : CArray
            Array to be used for training.
            Shape of input array depends on the algorithm itself.
        y : CArray
            Flat array with the label of each pattern.
            Can be None if not required by the algorithm.

        Returns
        -------
        CModule
            Trained instance of CModule.

        """
        x, y = self._check_input(x, y)
        self._clear_cache()

        if self.preprocess is not None:
            x = self.preprocess.fit_forward(x, y)

        return self._fit(x, y)

    _fit.__doc__ = fit.__doc__  # Same doc for the protected method

    # TODO: make abstract or call _fit_forward
    def fit_forward(self, x, y=None, caching=False):
        """Fit estimator using data and then execute forward on the data.

        This method is equivalent to call fit(data) and forward(data)
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
        forward : run forward function on input data.

        """
        # TODO: this is inefficient: it's going twice through pre-processing
        self.fit(x, y)  # train preprocessor chain first
        return self.forward(x, caching=caching)  # fwd again through the chain

    def gradient(self, x, w=None):
        """Compute gradient at x by doing a backward pass.

        Input will be preprocessed first and pre-multiplied by w if provided.

        """
        # Transform data using inner preprocess, if defined
        x, y = self._check_input(x)
        self._check_is_fitted()
        self._clear_cache()

        x_prc = self._forward_preprocess(x, caching=True)
        if self._grad_requires_forward:
            self._forward(x_prc)  # this is called only if required
        return self.backward(w)
