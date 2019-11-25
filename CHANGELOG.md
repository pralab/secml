## v0.10 (29/10/2019)
- #535 Added new package `secml.explanation`, which provides different methods for explaining machine learning models. See documentation and examples for more information.
- #584 **[beta]** Added `CAttackEvasionCleverhans` to support adversarial attacks from [CleverHans](https://github.com/tensorflow/cleverhans), a Python library to benchmark vulnerability of machine learning systems to adversarial examples.

### Requirements (1 change)
- #580 PyTorch version `1.3` is now supported.

### Added (4 changes)
- #565 Added new abstract interface `CClassifierDNN` from which new classes implementing Deep Neural Networks can inherit.
- #555 Added `CNormalizerDNN`, which allows using a `CClassifierDNN` as a preprocessor.
- #593 Added `CDataLoaderTorchDataset`, which allows converting a `torchvision` dataset into a `CDataset`.
- #598 Added gradient method for `CKernelHistIntersection`.

### Improved (6 changes)
- #562 Extended support of `CClassifierPyTorch` to nested PyTorch modules.
- #594 `CClassifierPyTorch.load_model()` is now able to also load models trained with PyTorch (without using our wrapper). New parameter `classes` added to the method to match classes to indexes in the loaded model.
- #579 Left side single row/column broadcast is now supported for sparse vs sparse `CArray` operations.
- #582 Improved performance of `CNormalizerMeanStd` when multiple channels are defined.
- #576 Vastly improved the performance of kernels by removing loops over samples in many classes and refactoring main routines.
- #562 Improved `grad_f_x` computation at a specific layer in `CClassifierPyTorch`.

### Changed (4 changes)
- #578 `CClassifierPyTorch` now inherits from `CClassifierDNN`. The following changed accordingly: parameter `torch_model` renamed to `model`; property `layer_shapes` is now defined; method `save_checkpoint` removed.
- #562 Parameter `layer` of `CClassifierPyTorch.get_layer_output()` is now renamed `layer_names` as a list of layers names is supported (a single layer name is still supported as input). A dictionary is returned if multiple layers are requested. See the documentation for more information.
- #533 Double initialization in `CAttackEvasionPGDLS` will now be executed regardless of the classifier type (linear or nonlinear) if the `double_init` parameter of `.run()` method is set to `True`.
- #591 It is now not required to call the `fit` method of `CNormalizerMeanSTD` if fixed mean/std values are used.

### Fixed (4 changes)
- #561 Fixed `CConstraintBox` not always applied correctly for float data.
- #577 Fixed `CClassifierPyTorch.decision_function` applying preprocess twice.
- #581 Fixed gradient computation of `CKernelChebyshevDistance`.
- #599 Kernels using distances are now based on negative distances (to correctly represent similarity measures). Affected classes are: `CKernelChebyshevDistance`, `CKernelEuclidean`.

### Removed & Deprecated (5 changes)
- #561 Removed parameter `precision` from `CConstraint.is_violated()`.
- #575 Parameter `batch_size` of `CKernel` is now deprecated.
- #597 Removed unused parameter `gamma` from `CKernelChebyshevDistance`.
- #596 Removed `CKernelHamming`.
- #602 Renamed `CNormalizerMeanSTD` to `CNormalizerMeanStd`. The old class has been deprecated and will be removed in a future vearsion.

### Documentation (5 changes)
- #538 Added a notebook tutorial on the use of Explainable ML methods provided by the `secml.explanation` package.
- #573 Improved visualization of attack results in `07-ImageNet` tutorial.
- #610 Fixed spacing between parameter and parameter type in the docs.
- #605 Fixed documentation of classes requiring extra components not being displayed.
- #608 Added acknowledgments to `README`.


## v0.9 (11/10/2019)
- #536 Added `CClassifierPytorch` to support Neural Networks (NNs) through [PyTorch](https://pytorch.org/) deep learning platform.

### Improved (1 change)
- #556 `CFigure.imshow` now supports `PIL` images as input.

### Changed (1 change)
- #532 Method `CPreProcess.revert()` is now renamed `.inverse_transform()`.

### Fixed (1 change)
- #554 Fixed `CClassifierSkLearn.predict()` not working when using pretrained sklearn models.

### Documentation (2 changes)
- #559 Deprecated functions and classes are now correctly visualized in the documentation.
- #560 Updated development roadmap accordingly to `0.10`, `0.11` and `0.12` releases.

### Deprecations (3 changes)
- #532 Method `CPreProcess.revert()` is deprecated. Use `.inverse_transform()` instead.
- #552 `CClassifierKDE` is now deprecated. Use `CClassifierSkLearn` with `sklearn.neighbors.KernelDensity` instead.
- #553 `CClassifierMCSLinear` is now deprecated. Use `CClassifierSkLearn` with `sklearn.ensemble.BaggingClassifier` instead.


## v0.8.1 (05/09/2019)
This version does not contain any significant change.

### Documentation (2 changes)
- #523 Fixed documentation not compiling under Sphinx v2.2.
- #529 Updated roadmap accordingly for v0.9 release.


## v0.8 (06/08/2019)
- First public release!