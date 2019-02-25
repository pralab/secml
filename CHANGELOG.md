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
- #292 Default configuration file `secml-lib.conf` is now copied to `$SECML_HOME_DIR` if not already there.

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
- Renamed settings.txt file to secml-lib.conf
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
