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