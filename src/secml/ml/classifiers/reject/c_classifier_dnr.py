"""
.. module:: CClassifierDNR
   :synopsis: Deep Neural Rejection (DNR) Classifier

.. moduleauthor:: Angelo Sotgiu <angelo.sotgiu@unica.it>

"""
from secml.array import CArray
from secml.ml import CClassifier
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.classifiers import CClassifierDNN
from secml.ml.features.normalization import CNormalizerDNN
from secml.core.exceptions import NotFittedError
from secml.core.attr_utils import add_readwrite


class CClassifierDNR(CClassifierRejectThreshold):
    """Deep Neural Rejection (DNR) Classifier. It is composed by a wrapped
    DNN, one or more layer classifiers trained on inner DNN layer outputs and
    a combiner trained on layer classifiers scores.

    DNR analyzes the representations of input samples at different network
    layers, and rejects samples which exhibit anomalous behavior with respect
    to that observed from the training data at such layers.

    More details can be found in `tutorials/12-DNR.ipynb` and in:
     - https://arxiv.org/pdf/1910.00470.pdf, EURASIP JIS 2020.

    Parameters
    ----------
    combiner : CClassifier
        The output classifier of DNR. It is trained on layer classifier scores.
        Its output is thresholded in order to reject samples.
    layer_clf : CClassifier or dict
        Layer classifier, trained on DNN inner layers outputs.
        If CClassifier, it is cloned for each selected layer.
        If dict, it must contain an item for each selected layer as
        `{'layer_name': CClassifier}`.
    dnn : CClassifierDNN
        An already trained DNN to be defended.
    layers : list of str
        Name of one or more DNN layers which outputs will be used to train
        layer classifiers.
    threshold : float
        The reject threshold applied to the combiner outputs. If the maximum
        class score of the combiner is lower than the threshold, the sample is
        rejected.
    n_jobs : int, optional
        Number of parallel workers to use for training the classifier.
        Cannot be higher than processor's number of cores. Default is 1.
    """
    __class_type = 'dnr'

    def __init__(self, combiner, layer_clf, dnn, layers, threshold, n_jobs=1):

        self.n_jobs = n_jobs
        super(CClassifierDNR, self).__init__(combiner, threshold)

        if not isinstance(dnn, CClassifierDNN):
            raise TypeError("`dnn` must be an instance of `CClassifierDNN`")
        if not isinstance(layers, list):
            raise TypeError("`layers` must be a list")
        if isinstance(layer_clf, dict):
            if not sorted(layers) == sorted(layer_clf.keys()):
                raise ValueError("`layer_clf` dict must contain `layers` "
                                 "values as keys")
            if not all(isinstance(c, CClassifier) for c in layer_clf.values()):
                raise TypeError("`layer_clf` dict must contain `CClassifier` "
                                "instances as values")
        elif not isinstance(layer_clf, CClassifier):
            raise TypeError("`layer_clf` must be an instance of either"
                            "`CClassifier` or `dict`")

        self._layers = layers
        self._layer_clfs = {}
        for layer in self._layers:
            if isinstance(layer_clf, dict):
                self._layer_clfs[layer] = layer_clf[layer]
            else:
                self._layer_clfs[layer] = layer_clf.deepcopy()
            # search for nested preprocess modules until the inner is reached
            module = self._layer_clfs[layer]
            while module.preprocess is not None:
                module = module.preprocess
            # once the inner preprocess is found, append the dnn to it
            module.preprocess = CNormalizerDNN(net=dnn, out_layer=layer)
            # this allows to access inner classifiers using the
            # respective layer name
            add_readwrite(self, layer, self._layer_clfs[layer])

    def _fit(self, x, y):
        """Extract the scores from layer classifiers and train the combiner.

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features).
        y : CArray
            Array of shape (n_samples,) containing the class labels.

        Returns
        -------
        trained_clf : CClassifier
            Instance of the classifier trained using input dataset.

        """
        x = self._create_scores_dataset(x, y)
        self._clf.fit(x, y)
        return self

    def _create_scores_dataset(self, x, y):
        """Extract validation scores from layer classifiers using the
        `fit_forward` function, based on a cross validation

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features).
        y : CArray
            Array of shape (n_samples,) containing the class labels.

        Returns
        -------
        concat_scores : CArray
            Concatenated layer classifiers scores, shape
            (n_samples, n_classes - 1), where n_classes includes
            the reject class.

        See Also
        --------
        :meth:`.CClassifier.fit_forward`

        """
        n_classes = y.unique().size
        # array that contains concatenate scores of layer classifiers
        concat_scores = CArray.zeros(
            shape=(x.shape[0], n_classes * len(self._layers)))

        for i, layer in enumerate(self._layers):
            scores = self._layer_clfs[layer].fit_forward(x, y)
            concat_scores[:, i * n_classes: n_classes + i * n_classes] = scores
        return concat_scores

    def _get_layer_clfs_scores(self, x):
        """

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).

        Returns
        -------
        concat_scores : CArray
            Concatenated layer classifiers scores, shape
            (n_samples, n_classes - 1), where n_classes includes
            the reject class.

        """
        caching = self._cached_x is not None
        n_classes = self.n_classes - 1
        # array that contains concatenate scores of layer classifiers
        concat_scores = CArray.zeros(
            shape=(x.shape[0], n_classes * len(self._layers)))

        for i, l in enumerate(self._layers):
            scores = self._layer_clfs[l].forward(x, caching=caching)
            concat_scores[:, i * n_classes: n_classes + i * n_classes] = scores
        return concat_scores

    def _forward(self, x):
        """"Private method that computes the decision function.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).

        Returns
        -------
        scores : CArray
            Array of shape (n_patterns, n_classes) with classification
            score of each test pattern with respect to each training class.
            Will be returned only if `return_decision_function` is True.

        """
        caching = self._cached_x is not None

        layer_clfs_scores = self._get_layer_clfs_scores(x)
        scores = self._clf.forward(layer_clfs_scores, caching=caching)

        # augment score matrix with reject class scores
        rej_scores = CArray.ones(x.shape[0]) * self.threshold
        scores = scores.append(rej_scores.T, axis=1)
        return scores

    def _backward(self, w):
        """Compute the gradient of the classifier output wrt input.

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
        grad = CArray.zeros(self.n_features)

        # the derivative w.r.t. the rejection class is zero, thus we can just
        # call the combiner gradient by removing the last element from w.
        grad_combiner = self._clf.backward(w[:-1])

        n_classes = self.n_classes - 1
        for i, l in enumerate(self._layers):
            # backward pass to layer clfs of their respective w
            grad += self._layer_clfs[l].backward(
                w=grad_combiner[i * n_classes: i * n_classes + n_classes])
        return grad

    @property
    def _grad_requires_forward(self):
        """Returns True if gradient requires calling forward besides just
        computing the pre-processed input x. This is useful for modules that
        use auto-differentiation, like PyTorch, or if caching is required
        during the forward step (e.g., in exponential kernels).
        It is False by default for modules in this library, as we compute
        gradients analytically and only require the pre-processed input x."""
        return True
