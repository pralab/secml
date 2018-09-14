from sklearn.neural_network import MLPClassifier
from secml.classifiers import CClassifier
from secml.array import CArray
from sklearn.utils.extmath import safe_sparse_dot
import warnings
import copy
import numpy as np
from sklearn.neural_network._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from secml.array.data_coversion_utils import nd_list_to_carray, flat_carray_to_np_list
from secml.core.constants import inf

#fix training starting
# todo: using only CArray in loss and gradients computation functions
class CClassifierMLP(CClassifier):
    """Multiple Layer Perceptron classifiers."""
    class_type = 'mlp'

    def __init__(self, n_classes=2, hidden_layer_sizes=(100,), activation='relu', solver='sgd', alpha=0.0001,
                 batch_size='auto',
                 learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                 normalizer=None):
        # Calling the superclass init
        CClassifier.__init__(self, normalizer=normalizer)

        # Classifier parameters
        self._n_classes = n_classes
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._solver = solver
        self._alpha = alpha
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._learning_rate_init = learning_rate_init
        self._power_t = power_t
        self._max_iter = max_iter
        self._shuffle = shuffle
        self._random_state = random_state
        self._warm_start = warm_start
        self._momentum = momentum
        self._nesterovs_momentum = nesterovs_momentum
        self._early_stopping = early_stopping
        self._validation_fraction = validation_fraction
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon

        self._trained = False

        self._loss = "log_loss"

        # Setting up clf parameters
        self._mlp = MLPClassifier(hidden_layer_sizes=self._hidden_layer_sizes, activation=self._activation,
                                  solver=self._solver, alpha=self._alpha,
                                  batch_size=self._batch_size,
                                  learning_rate=self._learning_rate, learning_rate_init=self._learning_rate_init,
                                  power_t=self._power_t, max_iter=self._max_iter,
                                  shuffle=self._shuffle,
                                  random_state=self._random_state, tol=1e-12, verbose=False,
                                  warm_start=self._warm_start,
                                  momentum=self._momentum,
                                  nesterovs_momentum=self._nesterovs_momentum,
                                  early_stopping=self._early_stopping, validation_fraction=self._validation_fraction,
                                  beta_1=self._beta_1, beta_2=self._beta_2,
                                  epsilon=self._epsilon)

    # fixme: la clear dovrebbe resettare self._trained, solo che viene chiamata durante il training quindi non possiamo farlo
    def __clear(self):
        """Reset the object."""
        pass

    def is_clear(self):
        """Returns True if object is clear."""
        return not self._trained

    @property
    def w(self):
        return nd_list_to_carray(self._mlp.coefs_)

    @property
    def b(self):
        return nd_list_to_carray(self._mlp.intercepts_)

    @w.setter
    def w(self, value):
        w_np_lists = self._mlp.coefs_
        self._mlp.coefs_ = flat_carray_to_np_list(value, w_np_lists)

    @b.setter
    def b(self, value):
        b_np_lists = self._mlp.intercepts_
        self._mlp.intercepts_ = flat_carray_to_np_list(value, b_np_lists)

    def _train(self, dataset):
        """Trains the MLP clf.

        The following is a private method for training the clf

        Parameters
        ----------
        dataset : CDataset
            Binary (2-classes) training set. Must be a :class:`.CDataset`
            instance with patterns data and corresponding labels.

        Returns
        -------
        trained_cls : clf
            Instance of the used solver trained using input dataset.

        """
        X, Y = self._data_preproc(dataset.X, dataset.Y)

        self._mlp.fit(X, Y)
        self._trained = True

        self._initial_w = None
        self._initial_b = None

        return self

    def _discriminant_function(self, x, label=1):
        """Compute the distance of the samples in x from the separating hyperplane.

        Discriminant function is always computed wrt positive class.

        Parameters
        ----------
        x : CArray or array_like
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        label : int
            The label of the class with respect to which the function
            should be calculated.

        Returns
        -------
        score : CArray or scalar
            Flat array of shape (n_patterns,) with discriminant function
            value of each test pattern or scalar if n_patterns == 1.

        """
        return CArray(CArray(self._mlp.predict_proba(x.tondarray()))[:, label])

    def discriminant_function(self, x, label=1):
        """Compute the distance of the samples in x from the separating hyperplane.

        Discriminant function is always computed wrt positive class.

        Parameters
        ----------
        x : CArray or array_like
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        label : int
            The label of the class with respect to which the function
            should be calculated.

        Returns
        -------
        score : CArray or scalar
            Flat array of shape (n_patterns,) with discriminant function
            value of each test pattern or scalar if n_patterns == 1.

        """
        return super(CClassifierMLP, self).discriminant_function(x, label)

    def _data_preproc(self, X, y=None):
        """
        Transform data and labels in numpy array and ensure that y are 2D
        :return:
        """
        X = X.tondarray()
        X = np.atleast_2d(X)

        if not isinstance(y, CArray):
            y = CArray(y)
        y = y.tondarray()

        new_y = CArray.zeros((X.shape[0], self._n_classes)).tondarray()
        new_y[CArray.arange(0, X.shape[0]).tolist(), y.tolist()] = 1

        return X, new_y

    def _activation_computations(self, X, layer_units):
        """
        Perform the forward step computing the neuron activation values

        :return:
        """
        n_samples = X.shape[0]

        # lbfgs does not support mini-batches
        if self._solver == 'lbfgs':
            batch_size = n_samples
        elif self._batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            if self._batch_size < 1 or self._batch_size > n_samples:
                warnings.warn("Got `batch_size` less than 1 or larger than "
                              "sample size. It is going to be clipped")
            batch_size = np.clip(self._batch_size, 1, n_samples)

        # Initialize lists
        activations = [X]
        activations.extend(np.empty((batch_size, n_fan_out))
                           for n_fan_out in layer_units[1:])

        # Forward propagate
        activations = self._mlp._forward_pass(activations)

        return activations

    def loss(self, X, y):
        """
        Compute network loss
        :param x:
        :param y:
        :return:
        """
        # convert data in the format used from sklearn
        X, y = self._data_preproc(X, y)

        n_samples, n_features = X.shape
        n_outputs = self._n_classes

        layer_units = ([n_features] + list(self._hidden_layer_sizes) +
                       [n_outputs])

        activations = self._activation_computations(X, layer_units)

        n_samples = X.shape[0]

        # get loss
        loss_func_name = self._loss
        if loss_func_name == 'log_loss' and self._mlp.out_activation_ == 'logistic':
            loss_func_name = 'binary_log_loss'
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1])
        # add l2 regularization term to loss
        values = np.sum(
            np.array([np.dot(s.ravel(), s.ravel()) for s in self._mlp.coefs_]))
        loss += (0.5 * self._mlp.alpha) * values / n_samples

        return loss

    def cmpt_lyr_output(self, X, lyr_idx):
        """
        Compute network loss
        :param x:
        :param y:
        :return:
        """
        # convert data in the format used from sklearn
        X = X.tondarray()
        X = np.atleast_2d(X)

        n_samples, n_features = X.shape
        n_outputs = self._n_classes

        layer_units = ([n_features] + list(self._hidden_layer_sizes) +
                       [n_outputs])

        activations = self._activation_computations(X, layer_units)

        return CArray(activations[lyr_idx-1])

    ################# loss grad wrt training parameters
    def gradient_loss_W(self, X, y):
        return self.gradient_loss_trparams(X, y)[0].ravel()

    def gradient_loss_b(self, X, y):
        return self.gradient_loss_trparams(X, y)[1].ravel()

    def gradient_loss_trparams(self, X, y):
        """
        Compute the gradient of the loss respect to each network weight
        :param X:
        :param y:
        :return:
        """
        X, y = self._data_preproc(X, y)

        n_samples, n_features = X.shape
        n_outputs = self._n_classes
        layer_units = ([n_features] + list(self._hidden_layer_sizes) +
                       [n_outputs])

        activations = self._activation_computations(X, layer_units)

        # initialize the data structures needed for gradient computation
        deltas = [np.empty_like(a_layer) for a_layer in activations]

        coef_grads = [np.empty((n_fan_in_, n_fan_out_)) for n_fan_in_,
                                                            n_fan_out_ in zip(layer_units[:-1],
                                                                              layer_units[1:])]

        intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in
                           layer_units[1:]]

        n_samples = X.shape[0]

        # Backward propagate
        last = self._mlp.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        #        print "initial delta last ", deltas[last]
        deltas[last] = activations[-1] - y
        #        print "activations-1 ", activations[-1]
        #        print "y ", y
        #        print "deltas last ", deltas[last]

        # Compute gradient for the last layer
        coef_grads, intercept_grads = self._compute_one_lyr_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads)

        # Iterate over the hidden layers
        for i in range(self._mlp.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self._mlp.coefs_[i].T)
            inplace_derivative = DERIVATIVES[self._activation]
            inplace_derivative(activations[i], deltas[i - 1])

            coef_grads, intercept_grads = self._compute_one_lyr_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads,
                intercept_grads)

        return nd_list_to_carray(coef_grads), nd_list_to_carray(intercept_grads)

    def _compute_one_lyr_loss_grad(self, layer, n_samples, activations, deltas,
                                   coef_grads, intercept_grads):
        """Compute the gradient of loss with respect to coefs and intercept for
        specified layer.

        This function does backpropagation for the specified one layer.
        """
        # outher product
        coef_grads[layer] = safe_sparse_dot(activations[layer].T,
                                            deltas[layer])
        coef_grads[layer] += (self._mlp.alpha * self._mlp.coefs_[layer])
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = np.mean(deltas[layer], 0)

        return coef_grads, intercept_grads

    ########## compute loss grad wrt x
    def gradient_loss_x(self, X, y):
        """
        Compute the gradient of the loss respect to the network input

        :param X: single sample features
        :param y: single sample label
        :return:
        """
        X, y = self._data_preproc(X, y)

        n_samples, n_features = X.shape
        n_outputs = self._n_classes
        layer_units = ([n_features] + list(self._hidden_layer_sizes) +
                       [n_outputs])

        activations = self._activation_computations(X, layer_units)

        # initialize the data structures needed for gradient computation
        deltas = [np.empty_like(a_layer) for a_layer in activations]

        # Backward propagate
        last = self._mlp.n_layers_ - 2

        #######################################compute deltas as they are needed for the gradient computation

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = activations[-1] - y

        # Iterate over the hidden layers
        for i in range(self._mlp.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self._mlp.coefs_[i].T)
            inplace_derivative = DERIVATIVES[self._activation]
            inplace_derivative(activations[i], deltas[i - 1])

        ###################################### compute the gradient of the error wrt the input
        input_grad = safe_sparse_dot(deltas[0], self._mlp.coefs_[0].T)

        return CArray(input_grad).ravel()

    def _gradient_x(self, X, y=1):
        """Computes the gradient of the y-th output neuron wrt 'x'.

        Parameters
        ----------
        x : CArray
            Pattern with respect to which the gradient will be computed.
            Shape (1, n_features) or (n_features,).
        y : int, optional
            Index of the class wrt the gradient must be computed.
            Default is 1, corresponding to the positive class.

        Returns
        -------
        gradient : CArray
            Flat array with the gradient of SVM wrt input pattern.

        """
        orig_y = y  # y is an integer
        X, y = self._data_preproc(X, y)  # y has dimension (n_samples, n_classes)

        n_samples, n_features = X.shape
        n_outputs = self._n_classes
        layer_units = ([n_features] + list(self._hidden_layer_sizes) +
                       [n_outputs])

        activations = self._activation_computations(X, layer_units)

        # initialize the data structures needed for gradient computation
        deltas = [np.empty_like(a_layer) for a_layer in activations]

        # Backward propagate
        last = self._mlp.n_layers_ - 2

        #######################################compute deltas as they are needed for the gradient computation

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = copy.deepcopy(activations[-1])
        label = np.zeros((1, self._n_classes))
        label[0, 1] = activations[-1][0][orig_y]
        deltas[last] = label

        # Iterate over the hidden layers
        for i in range(self._mlp.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self._mlp.coefs_[i].T)
            inplace_derivative = DERIVATIVES[self._activation]
            inplace_derivative(activations[i], deltas[i - 1])

        ###################################### compute the gradient of the y-th output neuron wrt the input
        input_grad = safe_sparse_dot(deltas[0], self._mlp.coefs_[0].T)

        return CArray(input_grad).ravel()

    ################# loss network output wrt training parameters
    # fixme: vedere per multiclasse
    def gradient_f_W(self, X, y=1):
        X = X.atleast_2d()
        grad = CArray.zeros(shape=(X.shape[0], self.w.size))
        for idx in xrange(X.shape[0]):
            x = X[idx, :]
            grad[idx, :] = self.gradient_f_trparams(x, y)[0]
        return grad

    def gradient_f_b(self, X, y=1):
        X = X.atleast_2d()
        grad = CArray.zeros(shape=(X.shape[0], self.b.size))
        for idx in xrange(X.shape[0]):
            x = X[idx, :]
            grad[idx, :] = self.gradient_f_trparams(x, y)[1]
        return grad

    def gradient_f_trparams(self, X, y):
        """
        Compute the gradient of the loss respect to each network weight

        :param X:
        :param y:
        :return:
        """
        orig_y = y  # y is an integer
        X, y = self._data_preproc(X, y)

        n_samples, n_features = X.shape
        n_outputs = self._n_classes
        layer_units = ([n_features] + list(self._hidden_layer_sizes) +
                       [n_outputs])

        activations = self._activation_computations(X, layer_units)

        # initialize the data structures needed for gradient computation
        deltas = [np.empty_like(a_layer) for a_layer in activations]

        coef_grads = [np.empty((n_fan_in_, n_fan_out_)) for n_fan_in_,
                                                            n_fan_out_ in zip(layer_units[:-1],
                                                                              layer_units[1:])]

        intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in
                           layer_units[1:]]

        n_samples = X.shape[0]

        # Backward propagate
        last = self._mlp.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = copy.deepcopy(activations[-1])
        label = np.zeros((1, self._n_classes))
        label[0, 1] = activations[-1][0][orig_y]
        deltas[last] = label

        # Compute gradient for the last layer
        coef_grads, intercept_grads = self._compute_one_lyr_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads)

        # Iterate over the hidden layers
        for i in range(self._mlp.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self._mlp.coefs_[i].T)
            inplace_derivative = DERIVATIVES[self._activation]
            inplace_derivative(activations[i], deltas[i - 1])

            coef_grads, intercept_grads = self._compute_one_lyr_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads,
                intercept_grads)

        return nd_list_to_carray(coef_grads), nd_list_to_carray(intercept_grads)
