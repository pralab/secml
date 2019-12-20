CClassifier
===========

The unified interface :class:`CClassifier` defines the structure of classifiers.

We differentiate a standard classifier from Deep Neural Networks (DNNs),
which require a more advanced interface defined by :class:`CClassifierDNN`
(described below).

Standard classifiers (``CClassifier``)
--------------------------------------

List of methods to implement:

- ``_forward``: performs a forward pass of the input x.
  It should return the output of the decision function of the classifier.

- ``_backward``: this method returns the gradient of the decision function
  output with respect to data. It takes a ``CArray`` as input, ``w``,
  which pre-multiplies the gradient as in standard reverse-mode autodiff.

- ``_fit``: fit the classifier on the data. Takes as input a ``CDataset``.

DNN backends (``CClassifierDNN``)
---------------------------------

The backend for DNN (:class:`CClassifierDNN`) is based on the ``CClassifier``
interface, adding more methods specific to DNNs and their frameworks.

An example of how to extend the ``CClassifierDNN`` interface is our
PyTorch wrapper :class:`CClassifierPyTorch`.

List of methods to implement:

- ``_forward``: performs a forward pass of the input x. It is slightly
  different from the ``_forward`` method of ``CClassifier``, as it returns
  the output of the layer of the DNN specified in the attribute ``_out_layer``.
  If ``_out_layer`` is ``None``, the last layer output is returned (applies
  the softmax if ``softmax_outputs`` is True).

- ``_backward``: returns the gradient of the output of the DNN layer specified
  in ``_out_layer``, with respect to the input data.

- ``_fit``: trains the classifier. Takes as input a ``CDataset``.

- ``layers`` (property): returns a list of tuples containing the layers of the
  model, each tuple is structured as ``(layer_name, layer)``.

- ``layer_shapes`` (property): returns the output shape of each layer
  (as a dictionary with layer names as keys).

- ``_to_tensor``: converts a ``CArray`` into the tensor data type of the
  backend framework.

- ``_from_tensor``: converts a backend tensor  data type to a ``CArray``.

- ``save_model``: saves the model weight and  parameters into a gz archive.
  If possible, it should allow model restoring as a checkpoint, i.e. the user
  should be able to continue training of the restored model.

- ``load_model``: restores the model. If possible, it restores also the
  optimization parameters as the user may need to continue training.

It may be necessary to implement a custom data loader for the specific DNN
backend. The data loader should take as input a ``CDataset`` and load the data
for the backend. This is necessary because the inputs to the network may have
their own shapes, whereas the ``CArray`` treats each sample as a row vector.
We suggest to add the ``input_shape`` as an input parameter of the wrapper
and handle the conversion inside.
