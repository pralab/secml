## v0.7-dev (02/08/2019)
- #352 Added `CClassifierSkLearn`, a generic wrapper for any `scikit-learn` classifier.
- #76 Removed necessity to switch between subplot types in `CFigure`, using `.switch_sptype()` method, in order to get non-standard plot functions from `CPlot` subclasses (like `CPlotRoc`, etc.). All methods from `CPlot` subclasses are now available upon creation of a new subplot.
- #485 Added notebooks with tutorials for end user.

### Added (3 changes)
- #126 Added `data.data_utils.label_binarize_onehot` function which allow to encode a labels array to one-hot format.
- #486 Added new plotting function `plot_decision_regions` which plots decision boundaries and regions for the given classifier. Improved colormap support for `.plot_fun` and `.plot_ds`.
- #487 Added new plotting function `plot_contraint` which plots constraint bounds on bi-dimensional feature spaces.

### Improved (5 changes)
- #83 `CNormalizerMinMax` can now be used on sparse data.
- #36 Improved performance of `CClassifierDecisionTree`, `CClassifierKNN` and `CClassifierRandomForest` by classifying all input samples at once.
- #352 `CClassifier.decision_function` now accepts `None` as a value for the `y` parameter, which will make the function return the function wrt. all the classes.
- #352 `CClassifierNearestCentroid` is now natively multiclass.
- #451 Added ability to activate/deactivate double initialization when calling `CAttackEvasionPGDLS.run()`.

### Changed (10 changes)
- #364 `CDataset.get_labels_asbinary` is now `get_labels_ovr`, with parameter `pos_class` renamed to `pos_label` and always required. New separate method `CDataset.get_labels_onehot` to return the dataset labels in on-hot encoding.
- #483 `gradient-bls` is now the default solver for `CAttackPoisoning` and subclasses.
- #496 `secml.testing` is no more included in the library package.
- #493 Plotting method `.plot_fobj` renamed to `.plot_fun`.
- #492 Renamed method `CPlotMetric.plot_repetitions` to `.plot_roc_reps` and `CPlotMetric.plot_mean` to `.plot_roc_mean`.
- #467 `CPlotSecEval.plot_metric` is now `.plot_sec_eval`. Removed obsolete methods `CPlotSecEval.plot_metric_for_class`, `CPlotSecEval.compute_mean_metric` and `CPlotSecEval.compute_metric_at_param`.
- #167 Moved `plot_confusion_matrix` to new class `CPlotMetric`. The function now accepts true and predicted labels and internally generates the confusion matrix. Parameters normalize and colorbar are now False by default. Default colorbar is now 'Blues'. Merged `CPlotRoc` with `CPlotMetric`.
- #507 `COptimizerGrad` is now `COptimizerPGD` ('pgd') and `COptimizerGradBLS` is now `COptimizerPGDLS` ('pgd-ls'). Also updated evasion attacks accordingly. Added new class `CAttackEvasionPGD`.
- #508 Default for `CAttackPoisoning` is now to choose the random initialization points from the surrogate dataset instead of the validation dataset. Also renamed parameter `ts` of `CAttackPoisoning` to `val` in order to better explain its purpose.
- #164 `CDLRandomToy` has been split in specific loaders: `CDLIris`, `CDLDigits`, `CDLBoston`, `CDLDiabetes`.

### Fixed (10 changes)
- #484 `CArray.min`, `CArray.max`, `CArray.mean` now correctly return a dense array when computing values on a specific axis.
- #452 Fixed setitem operation of `CArray` when index is a boolean mask and input array has more than 1 dimension.
- #497 Filtering out warning introduced by `numpy` v1.17 raised by `scipy` v1.3 during set operation on lil matrices.
- #499 Fixed `DeprecationWarning` related to invalid escape sequence from `settings.py` under Python 3.7.
- #498 We now correctly import `MutableMapping` from `collections.abc` under Python 3 and from `collections` under Python 2.
- #84 Fixed `CClassifierMCSLinear` not storing the weights in sparse format when training dataset is sparse.
- #84 Fixed `CClassifierKDE` and `CClassifierReject` gradient not returning a sparse array when input is sparse.
- #503 Fixed computation of `sha1` on arrays having big shape.
- #504 Fixed kernel parameter of `CClassifierSGD` and `CClassifierRidge` not being settable.
- #451 Fixed `COtimizerPGDLS` not projecting on box bounds when using an `l2` constraint.

### Documentation (2 changes)
- #475 Improved documentation of `CArray.sort` method.
- #474 Updated README.md to reflect the version of `tensorflow` that will be installed (`1.14`).

### Removed (2 changes)
- #352 Temporarly removed parallelization from `CClassifier.predict` and `CClassifier.fit`. Will be re-added at a later stage.
- #323 The following packages have been removed and will be released at a later stage: `adv.defenses`, `explanation`, `ml.features.selection`, `optim.optimizers.learning_rate`, `pytorch`, `tf`.


## v0.6-dev (16/07/2019)
- #455 Multiple improvements and fixes related to the support of sparse data.
- #449 Internal refactoring of `COptimizer` and related classes, including faster processing of sparse data.
- #427 `CAttackEvasion` now defines a generic evasion attack problem. The evasion attack leveraging the Bisect Line Search solver, is now available as `CAttackEvasionBLS`.
- #426 Refactor of classifiers methods related to gradients. The following methods could now be available depending on the specific class: `grad_f_x` (replaces `gradient_f_x`), `grad_tr_params`, `grad_loss_params`, `grad_f_params`, `hessian_tr_params`. See documentation for more details.
- #323 Added new class `data.splitter.CChronologicalSplitter` which splits a dataset in train and test using a timestamp as split point. `python-dateutil` is now added as a library dependency to conveniently parse timestamps.
- #420 `CDataset` extra attributes are now stored in a separate **header** attribute inside a `CDatasetHeader` object. This allow easier indexing, cloning and merge of datasets without copying or losing the extra properties. Check docs for more informations.
- #423 Added a loader for the `iCubWorld28` dataset [[ref](https://robotology.github.io/iCubWorld)].
- #429, #420, #422 Added multiple new functions to `core` module.

### Requirements (1 change)
- #472 Minimum required version of `tensorflow` is now `1.14` in order to get latest updates and fix few `PendingDeprecationWarnings`.

### Added (5 changes)
- #429 Added new function `core.type_utils.is_bytes` to identify bytes strings. `to_builtin` function now converts numpy `bytes_` objects to built-in `bytes`.
- #422 Added new functions in `core.type_utils`: `is_inf`, `is_posinf`, `is_neginf`, `is_nan`.
- #420 Added new functions  in `core.attr_utils`: `as_protected`, `has_protected`, `add_readonly`, `add_readwrite`.
- #422 Added new methods to `CArray`: `.is_inf`, `.is_posinf`, `.is_neginf`, `.is_nan`.
- #407 Added wrappers of `warnings.catch_warnings` and `warnings.filterwarnings` in `CLog`.

### Improved (12 changes)
- #428 Improved performance of `COptimizerGradBLS` on sparse data.
- #464 Weights are now stored in sparse format if training data is sparse by `CClassifierLogistic`.
- #433 Using L2 constraint for discrete optimization is not supported in `COtimizerGradBLS`. Error is now properly raised in that case.
- #418 `CConstraintBox` is now always considered inactive if bounds contain at least one `inf`. Also, blocked computation of `.constraint()` in that case. A new formulation for `.is_violated` check is now used that is more reliable in presence of `inf`.
- #418 Improved input validation in `CConstraintBox`. If a bound is defined as scalar, will be equally applied to all dimensions. If defined as `CArray`, a different value for each dimension is expected.
- #471 The internal buffer of `CSparse` is now converted to lil format before a set operation for efficient changing of sparsity structure.
- #456 Multiplication between a sparse and a dense array now always return a sparse array (as the product does not alter the sparsity level).
- #447 `CArray.logical_and()` now avoids converting input sparse array to dense.
- #60 Improved performance of `CArray.comblist` by removing internal recursion.
- #439 `COptimizerGradBLS` is now directly available from the `optim.optimizers` package.
- #425 Removed unnecessary transformation to diag array in `CNormalizeLinear.gradient`.
- #421 `CDataset` now inherits from `CCreator`, allowing use of all of our global superclass functionalities like `.create()`, `.deepcopy()`, `.set_params` and more.

### Changed (2 changes)
- #449 `CLineSearch.minimize()` now replaces method `CLineSearch.line_search()`.
- #427 `CAttackEvasionCleverhans` now inherits from `CAttackEvasion`.

### Fixed (17 changes)
- #459 Fixed `CTrainTestSplit` not actually using random seed (if set).
- #461 Fixed `CDatasSplitterKfold`, `CDatasSplitterStratKFold` and `CDatasSplitterShuffle` converting sparse datasets to dense before splitting them.
- #443 Fixed list of lists assignment of sparse arrays when input is a 2-D dense array occurring in scipy v1.3+.
- #468 Fixed a bug preventing assignment of a single-item sparse `CArray` to a sparse `CArray` while using Scipy v1.3+.
- #465 Fixed indexing operations on sparse `CArray` not returning the correct result if slice step is different from 1.
- #457 Fixed `CArray.round()` and `CArray.pow()` methods not creating a copy of the original array if sparse.
- #463 Fixed `CClassifierLogistic` converting sparse training data to dense.
- #470 Fixed `COptimizerScipy` using `BFGS` method instead of `L-BFGS-B` method when bounds are defined.
- #441 Fixed `CClassifierRejectThreshold.n_features` property returning the number of training set features after preprocessing instead of before.
- #442 Fixed `CClassifierRejectDetector.n_features` property returning the number of training set features after preprocessing instead of before.
- #431 In case of discrete optimization in `COtimizerGradBLS`, the current descend direction is now correctly the sign of the gradient.
- #449 Fixed `COptimizerScipy.maximize()` optimizing the wrong function (without the inverted sign).
- #449 Fixed additional parameters of function and its gradient passed in `COptimizerGrad.minimize()` and `COptimizerGradBLS.minimize()`. Also fixed related behaviour of `COptimizer.maximize()`.
- #444 Fixed setup installing pre-release versions of dependencies.
- #366 Handled warnings related to matrix subclass using `scipy <= v1.3`.
- #430 `core.type_utils.is_str` now correctly recognizes numpy `unicode_` objects as strings.
- #419 Fixed `ConvergenceWarning` raised by `CClassifierLogistic` due to unnecessary custom initialization of weights.

### Removed (2 changes)
- #418 Removed setters for ub/lb attributes from `CConstraintBox`.
- #449 Removed unused parameter `discrete` from `COptimizer` and `COptimizerGrad`. Only our solver `COptimizerGradBLS` currently supports discrete optimization.


## v0.5-dev (29/03/2019)
- #198 `secml` now support Python >= 3.5. DEPRECATION: support for Python 2.7 will be dropped in a future release without advance notice.
- #347 The `clear` framework (`CCreator.clear()` and `.is_clear()` methods) is now removed. Alternative methods have been added to different classes, if necessary, to replace its functionality.
- #4 Updated and completed the PyTorch models zoo. We provide multiple widely-used pre-trained models for 'cifar10', 'cifar100' and 'imagenet' datasets. For a complete list of the available models see `secml.pytorch.models.PYTORCH_MODELS`.
- #70 Complete revision of `optimization` package, which is now called `optim`. All solvers are now subclasses of `COptimizer`, which now provides all the methods previously implemented as part of `CSolver`, including constraints, bounds and `.maximize()` method. The following solvers are available: `COptimizerScipy`, `COptimizerGrad` and `COptimizerGradBLS`. See documentation for more informations.

### Requirements (3 changes)
- #394 Raise minimum required version of `tensorflow` to `1.13.*` for compatibility with Python 3.7.
- #198 Raised minimum requirements in order to properly support Python 3.7 (coming soon): `numpy >= 1.15.*, scipy >= 1.2.*, scikit-learn >= 0.20.*, torch>=0.4.1`.
- #198 Temporarily added `six` as a required dependency as we use it for Python 2/3 compatibility. Will be removed in future versions.

### Python 3 support (23 changes)
- #398 Added parameter `encoding` (default 'ascii') to `dict_utils.load_dict`.
- #398 Updated `pickle_utils.load` to allow loading files pickled using Python 2 by adding parameter `encoding` (default 'bytes').
- #398 Fixed `CDataLoaderCIFAR` to allow loading the dataset under Python 3.
- #398 Multiple updates for Python 3 compatibility when opening external files.
- #393 Revised use of division (`/`) and floor division (`//`) operators to be compatible with Python 3.
- #395 Fixed call to deprecated `inspect.getargspec()` function.
- #392 Fixed calls to `dict.items()`, `dict.keys()` and `dict.items()` returning a view instead of a list in Python 3.
- #386 Fixed concatenation of strings and bytes in COptimizerScipy.
- #391 Revised any call to `range` to be compatible with Python 3.
- #388 Fixed use of `map` built-in function to be compatible with Python 3.
- #389 Fixed use of deprecated `.next()` method of iterators (Python 3 compatibility).
- #386 `core.type_utils` now uses type checks for `int` and `str` compatible with Python 3.
- #386 `VERSION` and `VERSION_REV` files are now stored using `ascii` encoding.
- #386 Use `ascii` to decode `requirements.txt` file.
- #386 Fixed `hashlib.sha1.update()` receiving standard strings as input in Python 3.
- #386 `CArray` is now stored and loaded from text files using `utf-8` encoding by default.
- #381 Updated all imports to be compatible with Python 3.
- #383 Update use of `zip` and `itertools.izip` function to be compatible with Python 3.
- #384 Removed any use of the deprecated methods of `dict` `.iterkeys(), .iteritems(), .itervalues(), .viewkeys(), .viewitems() .viewvalues()` (Python 3 compatibility).
- #382 Using `0o` as the prefix for octal literals (Python 3 compatibility).
- #379 Use `six.add_metaclass(...)` to replace `__metaclass__` deprecated in Python 3.
- #374 `print` is now only used as a function for Python 3 compatibility.
- #198 Updated setup to allow installation under Python 3.5+.

### Added (8 changes)
- #409 Added new method `CFunction.reset_eval()` which can be used to reset the function and gradient of function evaluations counters.
- #289 Added new configuration options to control logging functionality: `SECML_STORE_LOGS`, `SECML_LOGS_DIR`, `SECML_LOGS_FILENAME`, `SECML_LOGS_PATH`. By default, logs are now stored in the `SECML_HOME_DIR/logs` folder only if `SECML_STORE_LOGS` is active (default False). See docs for more information.
- #345 Added new extra `tf-gpu` to setup which allows installation of `tensorflow-gpu`.
- #360 Added `CFunctionMcCormick.bounds()` which returns the expected input domain of the function.
- #357 Added custom exception `NotFittedError` which will be raised if an object is used before training (before calling `.fit()`).
- #357 Added new utility function `util.mixed_utils.check_if_fitted` that checks if the input object is trained, by raising `NotFittedError` if not.
- #357 Added the new method `CClassifier.is_fitted()` that can be used to check if a classifier is trained.
- #347 Added a new `deprecated` decorator. Can be imported from the new `core.decorators` module.

### Improved (12 changes)
- #88 Output of `CConstraintL1` and `CConstraintL2` `.projection()` method is now correctly a sparse array if input is sparse.
- #247 Added ability to pass optional keyword arguments to `CFunction.approx_fprime` and `.check_grad`. This allows better control of parameters and the use of those method with function only accepting keyword arguments (and not also standard optional arguments).
- #293 Improved error messages of configuration files parser.
- #99 Bounds of `COptimizerScipy` are now passed to the `scipy.optimize` solvers if defined.
- #385 `CFunction` can now work with callables (functions) accepting matrices (arrays of multiple points).
- #369 Output of `COptimizerScipy` can now be controlled via the `.verbose` parameter.
- #368 Added new parameter `levels_linewidth` to `CPlot.plot_fobj` which controls the line width of the contour lines.
- #368 Added new parameters to `CPlot.plot_path`: `path_width`, to control the width of path line; `start_edgecolor` and `final_edgecolor` to control the color of the edge of the start and final point marker, respectively.
- #99 Added support for bounds (`CContraintBox`) in `CSolverScipy`.
- #362 Added a new stop condition to `COptimizerGrad` that checks if the function value has not changed between two subsequent iterations.
- #4 Added a new input parameter `ds_id` to `pytorch.models.dl_pytorch_model()` to download a specific model trained on different datasets.
- #357 Added new init parameter `pretrained` and corresponding internal attribute `_trained` to `CClassifierPyTorch`. This should be set to `True` if the input model is (or return) a pretrained PyTorch model.

### Changed (14 changes)
- #20 Removed `armijo-goldstein` criteria to stop search in `CLineSearchBisect`. Replaced with a simpler tolerance check.
- #351 Using zero or a boolean False as a power in case of sparse arrays is no more permitted. The array should be explicitly converted to dense if needed.
- #252 Moved `CUnitTest` to new package `secml.testing`. This can be used to create additional unittests for the library. `pytest` must be installed to use this package.
- #252 Requirements for unittesting (`pytest` and `pytest-cov` are no more installed by default as are not necessary for running the library. If needed for developing reasons, can be installed using the new extra component `unittests`.
- #377 `CClassifierCleverhans` is now `CModelCleverhans`.
- #360 Package `optimization` is now named `optim`.
- #360 `CSolver` is now `COptimizer`. As a result, all solvers are now subclasses of `COptimizer`. `COptimizer` now provides all the methods previosuly implemented as part of `CSolver`, including constraints, bounds and `.maximize()` method.
- #405 `CSolverGradDesc` is now `COptimizerGrad` as it implements a Gradient (descent/ascent) optimization method. `CSolverDescDir` is now `COptimizerGradBLS` as it implements the Gradient (descent/ascent) method with Bisect Line Search.
- #405 `optim.line_search` and `optim.learning_rate` packages are now moved inside `optim.optimizers` package.
- #358 Moved solvers from `adv.attacks.evasion.solvers` to the new package `optimization.optimizers`.
- #360 Created new solver type `COptimizerScipy` which wraps `scipy.optimize.minimize`.
- #360 Removed `.minimize()` from `COptimizer`. This is now part of `COptimizerScipy`.
- #360 Moved `COptimizer.check_grad()` and `COptimizer.approx_fprime()` to `CFunction`.
- #357 `CClassifierPyTorch.load_state()` will now avoid to estimate and set the parameters `classes` and `n_features` as we do not have reliable knowledge of those informations (the training dataset is unknown).

### Fixed (11 changes)
- #363 Fixed `CAttackEvasion` crashing when input data is sparse.
- #20 Fixed a bug in `CLineSearchBisect` where the optimization points were incorrectly casted to `int` even if the `eta` step is `float`.
- #397 Fixed `CDense.save` storing arrays of dtype `int` as `float`. An error will now be raised if the loaded file contains values that cannot be casted using the desired dtype.
- #344 Fixed output of `CArray.unique()` having `float` or `bool` dtype instead of `int` dtype in some occasions.
- #372 Fixed a bug in `CExploreDescentDirection` which lead to a crash if no bounds constraint is defined.
- #404 Fixed a bug in `CConstraintBox` which caused the constraint to not be applied correctly on arrays of `int` dtype.
- #401 Fixed a bug in `CConstraintL2` which caused `.gradient()` and `.projection()` methods to return an array of `nan` values if the norm of input is zero.
- #349 Fixed inability to add/subtract a zero scalar or a boolean `False` from a sparse `CArray`.
- #378 Fixed a crash of `CConstraintBox` when input data is a sparse array with no zero elements.
- #402 Removed use of deprecated and not correctly used `async` parameter while calling `.cuda()` in `.CClassifierPyTorch`.
- #400 Fixed setup not installing "post" versions of PyTorch when available.

### Removed (6 change)
- #396 Removed ability to pass a `slice` to parameter `usecols` of `CArray.load()`.
- #93 Removed support for multipoint input in `CConstraint` and subclasses. Use `CArray.apply_along_axis` if needed.
- #48 Removed attribute `CClassifierMulticlass.binary_classifiers`. The binary classifiers trained by the multiclass estimators are only intended to be used internally.
- #289 Removed ability to import configuration file `SECML_CONF` from main module `secml`. Use `secml.settings.SECML_CONF` instead.
- #326 Removed redundant calls to `.nan_to_num()` from `CNormalizer` and subclasses.
- #357 Removed `RuntimeError` raised by `CClassifierPyTorch.load_state()` if `input_shape` is `None`. If `input_shape` is unknown or `None`, input will not be reshaped before passing it to the network.


## v0.4-dev (08/03/2019)
- #329 The project is now officially called `SecML`.
- #210 Added new package `tf.clvhs` which provides a wrapper for [CleverHans](https://github.com/tensorflow/cleverhans).
- #261 Complete revision of data preprocessing routines in `CClassifier` and subclasses. Added new superclass `CPreProcess` which provides an interface for all data transformation methods, like feature normalizers and feature reducers.
- #174 `torch` (**PyTorch**) version `>=1.0.*` is now officially supported.
- Improvements to release process.

### Added (7 changes)
- #261 Added new superclass `CPreProcess` which provides an interface for all data transformation methods, like feature normalizers and feature reducers. `CNormalizer` and `CReducer` now inherit from `CPreProcess`.
- #261 Normalizers now accept `w` as a backpropagation vector to be left-multiplied to their gradient.
- #261 All `CPreProcess` subclasses now accept `preprocess` as an optional init parameter.
- #333 Added parameter `random_state` to `CClassifierRandomForest`.
- #264 Added `CClassifierPyTorchCNNMNIST` implementing the CNN model for MNIST digits dataset from N. Carlini and D. A. Wagner, "Adversarial examples are not easily detected: Bypassing ten detection methods", 10th ACM Workshop on Artificial Intelligence and Security. ACM, 2017.
- #210 Added new package `tf.clvhs` which provides a wrapper for [CleverHans](https://github.com/tensorflow/cleverhans), a Python library to benchmark machine learning systems' vulnerability to adversarial examples. In this first iteration, it is included the class `CCleverhansAttack`, which wraps the Cleverhans attacks in a `CAttackEvasion` object. For a list of all supported Cleverhans attacks, see `c_attack_evasion_cleverhans.SUPPORTED_ATTACKS`.
- #210 As preliminary support for `tensorflow` library, added different function to convert SecML classifiers' methods to `tensorflow` PyFunctions.

### Improved (3 changes)
- #261 Trainin data is now preprocessed (if a `preprocess` is defined) in `CClassifierPyTorch.fit` if no `train_transform` is defined.
- #266 Output of `CNormalizerPyTorch.gradient` method is now always a raveled vector.
- #280 Revised preprocess application in `CClassifierRejectThreshold` and `CClassifierRejectDetector` to make it consistent between fit, decision_function, predict, gradient.

### Changed (7 changes)
- #174 Raised minimum requirements for `torch` (**PyTorch**) to `>=0.4.*`.
- #329 The default paths of data and configuration files have changed accordingly. The new default `SECML_HOME_DIR` is `secml-data`. The new default name of the configuration file is `secml.conf`. Please follow the update guide to keep existing data while using the new version of the library.
- #230, #337 Renamed `CPca` and `CLda` to `CPCA` and `CLDA`, respectively.
- #261 `CClassifier` now expect a generic `CPreProcess` instance (or string with class type) as input to `preprocess` parameter.
- #266 `CClassifier` gradient is now backpropagated to preprocess inside `gradient_f_x`.
- #266 `y`, the label of the desired class wrt compute the gradient of the decision function, is now a required parameter of `CClassifier` and subclasses.
- #288 Renamed `normalize` and `fit_normalize` methods of `CNormalizer` to `transform` and `fit_transform`.

### Fixed (4 changes)
- #332 The value of the discriminant function of `CClassifierRejectThreshold` in the case of rejected samples is now correctly equal to the reject threshold. Also updated the gradient value accordingly.
- #327 In the case of indiscriminate evasion attack, the true class is now correctly excluded from the choice of the competing class (even if the sample is rejected).
- #290 Keyword raise is now correctly used in `CReducer` to raise errors instead of return.
- #334 Fixed data not being preprocessed in `CClassifierPyTorch.get_layer_output`.

### Removed (1 change)
- #261 Removed bugged LDA `revert` method.


## v0.3.1-dev (25/02/2019)
- Internal (improvements to release process)


## v0.3-dev (22/02/2019)
- #69 Added Poisoning attacks for SVM, Ridge and Logistic classifiers.
- #188 Implemented a framework for Explainable Machine Learning methods. Added explanation methods based on relevant features and relevant prototypes.
- #287 Added 2nd order gradients for linear, SVM, Ridge and Logistic classifiers as part of the new `ml.classifiers.gradients` framework.

### Added (12 changes)
- #69 Added `CAttackPoisoning` to perform training-time attacks. Added `svm`, `ridge` and `logistic` specializations.
- #287 Added `CClassifierGradient` to compute the gradients of the classifiers. Added specifications for `linear`, `svm`, `ridge`, `logistic`. All gradient functionality will be progressively moved to `CClassifierGradient` and subclasses.
- #291 Added `CExplainer` and `CExplainerLocal` abstract interfaces for Explainable ML methods.
- #291 Added `CExplainerLocalLinear` to explain linear classifier via their weights vector.
- #286 Added `CExplainerLocalInfluence` to compute the local explanations via relevant prototypes.
- #299 Added `CExplainerLocalIntegratedGradients` class which implements the Integrated Gradients method for local explanations (*Axiomatic Attribution for Deep Networks, Sundararajan et al., ICML 2017*).
- #314 Added `CClassifierLogistic` which implements Logistic Regression (aka logit, MaxEnt) classifier.
- #297 Added methods `is_linear` and `is_kernel_linear` to `CClassifierRidge`.
- #248 Added methods `is_linear` and `is_kernel_linear` to `CClassifierSGD`.
- #305 Added method `barh` to `CPlot` which produces a horizontal bar plot.
- #156 Added `CArray.sha1` method which computes the `sha1` hexadecimal hash of array.
- #304 Added method `CArray.get_nnz` which returns the number of non-zero elements along a given axis or the entire array.

### Improved (13 changes)
- #309 vector-like `CArray`s of sparse format and 2-D vector-like `CArray`s of dense format can now be used for indexing.
- #310 `CArray`s of sparse format can now be used to index a CArray of dense format.
- #192 Improved compatibility with `numpy` when using a list of lists as index to `CArray` (find-like indexing).
- #303 Now using `csr_matrix.nonzero` to return `CSparse.nnz_indices`.
- #279 Error is now raised in CClassifierMulticlass and subclasses if index y is outside [0, num_classifiers) range.
- #55 `CDataLoader` subclasses now acquire multiprocessing lock if they need to download/extract files.
- #280 Overriding preprocess getter/setter in `CClassifierRejectDetector` and `CClassifierRejectThreshold` to handle the preprocessor of the inner classifier.
- #280 Added missing properties `classes`, `n_classes` and `n_features` to CClassifierRejectDetector and changed fit function to avoid double setting `classes`, `n_classes`, `n_features` and `preprocess`.
- #282 Added a getter for the internal classifier and detector in `CClassifierRejectDetector`.
- #281 Added a setter/getter for the inner classifier in `CClassifierRejectThreshold`.
- #271 Added an option to the fit method of `CClassifierPyTorch` to decide whether or not to use the best parameters by accuracy score at the end of the training process.
- #253 `CDataLoaderImgFolders` and `CDataLoaderImgClients` now store the number of channels for each image.
- #292 Default configuration file `secml.conf` is now copied to `$SECML_HOME_DIR` if not already there.

### Changed (4 changes)
- #285 Moved `CClassifierMulticlass._gradient_f` implementation to `CClassifierMulticlassOVA` as it works only in case of OVA scheme.
- #155 The surrogate classifier passed to `CAttack` must now be differentiable (must implement `gradient_f_x`) and must be already trained on surrogate data.
- #304 `CArray.nnz` now calls respective `get_nnz` method with `axis=None`.
- #284 Renamed parameter `pos_label` of `CSoftmax.gradient` to `y`.

### Fixed (13 changes)
- #255 Fixed a bug of `is_intlike` and `is_floatlike` returning True for 0-sized arrays.
- #258 `type_utils.is_list_of_lists` now returns False for empty lists.
- #306 `CArray.nnz_data` now always returns a dense flat array as expected.
- #257 Empty lists used as indices are now explicitly converted to `ndarrays` of int dtype.
- #301 Fixed `CDLRandomToy` raising `TypeError` when `zero_one` parameter is set to True.
- #297 Fixed `CClassifierRidge.gradient_f_x` not taking into account the kernel function.
- #294 Fixed a crash in `CPlotRoc.plot_mean()` when a style is passed and `plot_std` is True.
- #290 Keyword raise is now used in `CReducer` to raise errors.
- #316 `CClassifierRejectThreshold.fit` now correctly returns an instance of itself.
- #283 `CClassifierRejectDetector.gradient_f_x` now takes into account the softmax function.
- #248 Fixed `CClassifierSGD.gradient_f_x` not taking into account the kernel function.
- #277 Now the alternative init point in `CAttackEvasion` is always computed using the correct `y_target`.
- #253 Fixed images loading in `CDataLoaderImgFolders` and `CDataLoaderImgClients` by raveling them on load.

### Documentation (1 change)
- #188 Updated `README` to introduce the `explanation` package.

### Removed (3 changes)
- #155 Removed automatic training of a SVM with RBF Kernel from `CAttack` if the surrogate classifier is not differentiable.
- #303 Removed `CSparse.nnz_row_indices` and `CSparse.nnz_column_indices` properties. Updated related methods to use `CSparse.nnz_indices` directly.
- #282 Removed redundant getters `classes`, `n_classes` and `n_features` from `CClassifierRejectDetector`.


## v0.2.2-dev (30/01/2019)

### Bugfix (4 changes)
- #262 Deepcopy of `CClassifierPyTorch` now correctly restores the parameters for both trained and untrained classifiers.
- #267 `start_epoch` in `CClassifierPyTorch` is now incremented on storing state instead of loading.
- #268 Accuracy and Best Accuracy are now reset when appropriate in `CClassifierPyTorch`.
- #269 `random_state` in `CClassifierPyTorch` is now called upon model initialization to allow reproducibility in case of retraining.


## v0.2.1-dev (28/01/2019)

### Bugfix (1 change)
- #254 Fixed not usable properties w and b of CClassifierPyTorch.


## v0.2-dev (23/01/2019)
- #183 Complete review of the `pytorch` package.
- #12 Classifiers with reject option based on a threshold.
- #182 Improvements to evasion attack while used on classifiers with reject option.
- Review of library requirements and compatibility with numpy, scipy, scikit-learn, and pytorch.

### New Features (12 changes)
- #12 Implemented a common interface for classifiers with reject option.
- #12 Implemented classifiers with reject option based on a threshold.
- #188 Implemented a defense based on a classifier with a detector inside.
- #213 Added an option to softmax-scale the outputs in CClassifierPyTorch.
- #205 Added random_state parameter to CClassifierPyTorch.
- #212 Create a model file for MLP
- #212 Added ability to load model params from file (limited to integer values).
- #220 Added ability to compute the gradient of CClassifierPyTorch at a specific layer.
- #217 Added CNormalizerPyTorch, normalizer with a PyTorch classifier inside.
- #234 Added the gradient of the softmax function
- #242 Added random_state parameter to CClassifierMCSLinear
- #57 Added randuniform classmethod for CArray, wrapper of np.random.uniform

### Improvements (13 changes)
- #121 Objective function returned by the run function of c_attack_evasion now return the mean over the points.
- #200 Improved logging of training phase of CClassifierPyTorch to use different verbosity levels.
- #199 We now use the best epoch and not the last in CClassifierPyTorch to set parameters.
- #212 Completely reviewed CClassifierPyTorch initialization and parameter.
- #220 Fixed and improved get_layer_output method of CClassifierPyTorch.
- #217 CClassifier gradient now manages normalizers accepting an accumulation vector.
- #202 Cleanup of parameters sorting in CTorchClassifier and renames.
- #202 Added properties for all params in CTorchClassifier and subclasses.
- #28 Updated and cleaned CNormalizerMeanStd.
- #139 The number of hidden layers in CTorchClassifierFC is now parametric.
- #27 Reviewed and updated eval utils file, now CMetricPyTorchAccuracy.
- #27 Review of pytorch.utils.misc module. Rename of secml.utils.fun_utils into mixed_utils.
- #181 Check if constraint is violated right after x is projected within the feasible domain. If the point remains outside, an exception is raised.

### Changed (17 changes)
- #206 Removed any reference to first_eva from CAttackEvasion
- #206 Removed CSecEvalDataEvasion as no more needed
- #196 Updated refs to CSparse and idx in CDense/CSparse.
- #174 Changed minimum requirements for torchvision to "torchvision>=0.1.8"
- #204 Removed softmax layer from mlp and using cross-entropy as loss (includes softmax itself)
- #202 batch_size is now a required parameter of CTorchClassifier.
- #197 Removed setters for w and b from CTorchClassifier.
- #28 Renamed pytorch.normalizers package to pytorch.normalization. Moved CNormalizerMeanSTD to main lib.
- #28 CNormalizerZScore is now removed and CNormalizerMeanStd can be used instead.
- #232 Moved settings module from core package to main package.
- #232 Moved PyTorch related settings to a separate settings module in pytorch package.
- #103 Added `pytorch` as extra requirement. Removed from requirements.txt
- #191 Blocked installation of numpy v1.17 or newer as they drop support for python 2.
- #193 Blocked installation of scipy v1.3 or newer as they drop support for python 2.
- #215 `PyTorch` is now after the superclass name in every pytorch class.
- #214 Rename CClassifierPyTorchFullyConnected to CClassifierPyTorchMLP.
- #216 Moved softmax function to a separate class CSoftmax.

### Bugfix (8 changes)
- #219 Fix training phase of CClassifierPyTorch broken after fit renaming.
- #235 Added missing calls to is_clear() in CClassifierPyTorch.predict/_decision_function
- #137 `weight_decay` is now applied only to weights if new `regularize_bias` init parameter is False.
- #138 Optimizer default parameters are now stored as part of the state dict.
- #182 Fixed implementation of the evasion objective function and relative gradient when reject classifier are used.
- #231 Small change in CPca for consistency with sklearn.
- #225 CClassifierPyTorch.load_state now updates CClassifier attributes.
- #223 Added a missing cast to item() in CEvasionAttack and CSolver to avoid f_obj and the idx of the current evasion point being shown as a CArray

### Documentation (1 change)
- #240 Update README to explain how to install the extras components

### Removed (1 change)
- #231 Removed CKernelPCA until further investigation of differences with sklearn.


## 0.1-dev (21/12/2018)
- Initial full development release

### Improvements (163 changes)
- #152 All class_type(s) are now defined as private class attributes
- create.attr_utils.as_private now can be called passing a private attribute directly.
- Updated README.md: added section on editable installations
- updated CArray unittests. Unittests for .all and .any largely improved.
- removed multiple no more necessary casts to CArray.
- CTorchDataset: labels are stored as 1-D or 2-D, keeping original shape. **getitem** return labels as 0-D tensors.
- Renamed samples to X and labels to Y inside CTorchDAtaset to follow our naming convention.
- Added check that input index is integer for CTorchDataset.**getitem**
- Real-time download progress for `download_utils.dl_file` is now printed only if stdout is a terminal to avoid spamming.
- default value for attack_classes is now 'all' meaning that all classes can be attacked.
- `CFunctionLinear` and `CFunctionQuadratic` do no more manage a multipoint input case, as it is not required.
- #85 Labels are now required in CDataset.
- #163 Now checking if number of labels is equal to number of samples in CDataset.
- #94 Added abstractproperty for clas_type attribute for all superclasses.
- #153 CCreator.create now accepts None as class type and returns an instance of the superclass.
- Added has_private and get_private functions to attr_utils to allow easier work with private attrs.
- Added "SECML_" prefix to each global variable to avoid name conflicts.
- secml home directory is now named secml-lib-data
- Renamed settings.txt file to secml.conf
- Small update to configuration file. Main section called "secml-lib"
- PYTORCH_DATA_DIR is now called `python-data`
- Added all definable parameters to configuration file.
- Config parse now accepts str as dtype
- added ability to parse configuration files from multiple locations.
- SECML_HOME_DIR is now created if not available.
- Added secml.config_fname() function to return the path of the active configuration file.
- Default config file moved to main package. Modified setup to account this.
- SECML_USE_CUDA is now SECML_PYTORCH_USE_CUDA as this parameters control pytorch wrapper only.
- Added _parse_env and _parse_env_config functions to allow reading global var from env or config files.
- #107 Added ability to specify an initial logging level for CLog.
- Most of the settings can now be specified from environment variables.
- Set default value for CLog propagate to False. Main loggers do not propagate, child should.
- CDense and CSparse classnames now use camelcase.
- Only CArray is now importable via "from secml.array import "
- Added `__floordiv__` and `__rfloordiv__` methods along with unittests.
- norm and norm_2d ord parameter is now `order` as `ord` is a built-in name.
- "sparse" is now the last keyword parameter of few CArray classmethod to make all of them uniform.
- Added few missing mathematical methods to CSparse. sin (available), cos, exp, log, log10, normpdf, interp (all not implemented)
- toarray method in CDense and CSparse renamed to tondarray to better represent its purpose
- binary_search is now a dense-only method.
- apply_fun_torow is now apply_along_axis as the optional axis parameter can be specified.
- Modified order of the parameters for CArray.load method
- argsort is now supposed to always return a CArray. Updated Csparse.argsort to support method parameter.
- maximum and minimum are now supposed to always return a CArray.
- return_counts extra param added to CArray.unique (dense and sparse). return_index available for sparse too.
- Bincount is now available for sparse arrays too.
- Multiple improvements for diag method. k parameter now working for sparse arrays too. ValueError is correctly raised for dense arrays if trying to extract a diagonal outside the array.
- For random generated methods, random_state is set in CDense/CSparse.
- LOTS of fixes/improvements to CArray docstrings.
- Creation of arrays with more than 2 dimensions is now blocked (temporarily).
- Added keepdims parameter to norm_2d method.
- Added a globals module to secml.
- Added CArrayInterface. Now CArray/CDense/CSparse inherit from it.
- Added flatten method to CDense and CSparse.
- CArray.sort method has now a inplace parameter. **DEFAULT FOR INPLACE IS NOW FALSE**
- Defined generic method rint, wrapper of round.
- Default value for density parameter of rand method is now 0.01 (consistent with scipy)
- Resorted and categorized methods in CArray, CDense and CSparse.
- Added dtype parameter to cumsum.
- Added parameter dtype to mean method.
- Checked/cleaned parameters and defaults of all CArrayInterface and subclasses methods
- CArray.empty now actually creates an empty array for sparse data.
- implemented all the available norms for sparse arrays.
- added norm_2d to CSparse to handle matricial norms and vector norms along axis.
- CArray.norm_2d now handle matricial norms and vector norms along axis.
- CArray.norm now ONLY handles vector norms for vector-like arrays.
- Updated docstring and parameters of CContraint.contraint method and subclasses.
- Updated CContraint.is_violated method so that correctly returns an array of booleans or a single bool. Added precision parameter.
- Now using modules from `sklearn.model_selection` instead of deprecated `sklearn.cross_validation`.
- Set the minimum `sklearn` version to `0.19.*`
- Relaxed strict requirement on sklearn version to reduce future maintenance.
- Updated import of sklearn LDA.
- Removed `optimization.contraints.CConstraintSet` class
- Reduced execution time of test_c_multi_ova by using less points in plot_fobj.
- Removed not needed verbosity level in test_plot_decision_function.
- Updated all uses of class_weight='auto' to class_weight='balanced'.
- Added new setting `pytorch.USE_CUDA`
- Added function that downloads pretrained pytorch models from our cloud
- #32 Multiple methods renamed. train -> fit; discriminat_function -> decision_function; classify -> predict
- #32 Predict method now only returns labels by default (like in sklearn)
- #114 Now each class must define a __is_clear private method which will be called by CCreator.is_clear.
- #10 Added CPlotStats from data_utils.
- #10 Removed no more needed module data_utils.
- #154 Implemented CTrainTestSplit utility function.
- renamed CMetricTPatFP to CMetricTPRatFPR
- fp_rate parameter in CMetricPAUC is now fpr.
- Updated tp/fp parameters to tpr/fpr when applicable in CRoc and related classes.
- #168 All applicable metrics now return a float value.
- #148 All class_types renamed accordingly to issue plan. Updated classes docstring.
- Working on #128: core.constants.inf and core.constants.nan are now wrappers of np.inf and np.nan
- #129 Using our constant where applicable instead of numpy ones.
- #122 Moved classifiers, features, kernel, peval, stats to the new ml package. Also updated affected imports.
- #110 extend_binary_labels is now convert_binary_labels
- reviewed CLoss and all subclassess
- CLossSquared (now CLossQuadratic #123). Added CLossSquare.
- #125: revamped CLossSoftMax, now CLossCrossEntropy. Added separate softmax function
- Added unittests for all CLoss classes
- CRowNormalizer is now CNormalizerUnitNorm.
- Updated some files to follow the correct naming convention.
- Modified check_binary_labels and extend_binary_labels to efficiently manage integer inputs.
- removed default value for label in discriminant function for CClassifierKNN, CClassifierRF, CClassifierDT
- Reviewed classify, discriminant_function and _discriminant_function methods for all classifiers.
- Added atleast_2D in discriminant_function and _discriminant_function for all classifiers.
- Updated and improved unittests for all classifiers
- Output of discriminant_function and classify is now converted to float in CTorchClassifier
- using scipy.norm.pdf instead of deprecated matplotlib.mlab.normpdf.
- Added max_iter and tol parameters replacing deprecated n_iter in SGD classifier. Currently we set tol default value to -np.inf, to keep the actual behavior. From sklearn 0.21 the default value will be 1e-3. We need to check it again in the future.
- set the minimum required version of scipy to 1.1.*
- Now using scipy implementation of reshape method.
- removed gradient method in favour of specific functions.
- Removed has_gradient method. A more pythonic way of check if the gradient is defined should be used, which is for example `try:  clf.gradient_f_x(..) expect NotImplementedError: "not differentiable"`
- Removed not used method _gradient_numerical_x
- Added function classifiers.clf_utils.check_binary_labels that checks if input labels are binary.
- Updated multiple classifier in order of adapt the gradient functions. CClassifierLinear, CClassifierSVM, CClassifierKDE, CClassifierMulticlass, CTorchClassifier
- Transformation should only be passed using the specific parameter in CTorchDataset.
- Removed possibility of passing a test_transformation in init. This could create problems with the gradient.
- Removed gradient_w_out_x from CTorchClassifier. A child classifier should be defined if necessary.
- CTorchClassifierDensenet is now CTorchClassifierDenseCifar.
- using setuptools.find_packages in setup.
- renamed torch_nn package to pytorch
- raised numpy minimum version requirement to 1.13
- cleanup of COptimizer interface, docs, properties.
- added opt_utils where fun_tondarray and fprime_tondarray are defined.
- COptimizer.solver setter is now removed. Solver should be passes as init param only.
- Added COptimizer.minimize method.
- added global_min, global_min_x, global_max, global_max_x methods in CFunction.
- Value of the objective function is now returned as scalar by COptimizer.minimize.
- Write a basic README for the project
- Working on #150: dev version should have a number, so currently we are at dev0
- VERSION now version.py which provides a more complex handling of version id.
- added VERSION file
- Using pkg_resources.parse_version to read and parse version string and be compliant with PEP440
- Improved setup with version reading and updated few parameters
- Moved docs to docs folder
- replaced pngmath deprecated sphinx extension with imgmath
- Using readTheDocs theme
- Added requirements file for docs.
- Updated matplotlib minimum requirement to 2.2 LTS
- Forcing maximum sklearn version to 0.20 as will be the latest supporting python2
- Added tox file to manage testing against multiple versions of the libraries.
- using x instead of data in feature.normalization classes as a more standard choice.
- #176 Normalizer parameter of CClassifier(s) renamed to preprocess.
- #165 CCustomTestMetric now defined inside test_perf_evaluator test.
- #56 Refactored CCreator.create() to allow only actual subclasses of a class to be created.
- #173 Added CDensityEstimation with basic unittests.
- Added random state param in SGD
- #166 class_type public property now raises error if __class_type is not defined.
- #15 Added basic installation guide.
- #150 Dev version should have a number, so currently we are at dev0
- #15 Using repo url for download_url
- Added install_requires option to setup in order to install dependencies.
- Added classifiers in setup
- Added platforms attribute to setuptools.
- #150 `__version__` now contains a string.
- Added test for gradient of classification losses.
- Changed class_type of log loss to `log` to be compatible with sklearn.
- CArray._instance_array is now _instance_data and a separate function to avoid the user use it.
- Added .item() method that returns the single element contained in the array as q built-in type.
- _check_index is now _prepare_idx to better represent its purpose.
- **getitem** now always returns a CArray.
- label parameter of (_)discriminant_function method is now y.
- #25 Removed openopt references.
- #117 Removed any numba-related file

### Bugfix (37 changes)
- #178 Setting zip_safe flag as False as we cannot support run from zip.
- #175 Fixed derivative of euclidean kernel. Added euclidean kernel to tests.
- #151 CAttackEvasion is now defined in adv.attacks package to allow use of .create()
- #116 Fixed CPlotRoc.plot_repetitions when a single roc rep is available.
- #170 Added a cast to float in CLossLogisticto avoid an unexpected conversion in int of the derivative
- #134 The correct value of workers to mimic n_jobs is now passed to num_workers param of torch.DataLoader
- #127 All CArray methods now return a CArray if axis parameter is not None.
- Fixed python_requires string in setup.
- pytorch models are mapped to cpu if cuda is not available on loading.
- Resolve "TypeError when using CLossSoftMax"
- CContraintBox now correctly manage ub/lb.
- `CContraint` now correctly defines `class_type` property.
- removed clipping of features outside feature_range in CNormalizerMinMax.
- output of CNormalizerMinMax is now correct even if min=max=0.
- Fixed use of `COptimizer` in multiple modules
- Fixed doctests of `COptimizer`
- #58: The format of the output of kernel gradient follow the format of the array wrt the kernel is computed.
- Added filter warning for deprecation introduced in sklearn version 0.20.1
- fixed multiple UserWarnings and RuntimeWarning related to matplotlib.
- unittests test files are now stored in the respectively test folder.
- passing only mean to plot in CPlotSecEval.plot_metric_for_class.
- score_exp and score_exp_sum now have compatible shape in CLossSoftMax.
- Fixed/cleaned CAttack.is_attack_class. Removed CAttack._is_attack_class method
- Cfunctions now always return a scalar as expected
- #49 Fixed `utils.download_utils.dl_file` crashes if username only auth is used.
- #133 CLossLogistic must return 1 for zero input
- Derivative of CLossHingeSquared was wrong.
- Fixed CClassifierKDE and CClassifierNC so that the output of discriminant_function is always CArray
- CTorchClassifierDenseNetCifar not correctly set the n_classes property.
- train of CTorchClassifier now correctly sets parameters and manage normalizer (by raising a warning).
- Corrected global minimum of CFunctionMCCormick
- #119 x,y are now passed as ravel to scatter function when c is an array.
- #75 Fixed unittests for perf evaluator and nans.
- Fixed gradient of cross entropy loss
- Fixed the cross entropy loss (removed the "pos_label" parameter from the loss function)
- Fixed the test of the gradient
- #132 Avoid float division by zero warnings in normalizers
