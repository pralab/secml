## v0.14.1 (22/04/2021)
- This version brings fixes for a few issues with the optimizers and related classes, along with improvements to documentation for all attacks, optimizers, and related classes.

### Fixed (3 changes)
- #923 Fixed `COptimizerPGDLS` and `COptimizerPGDLS` not working properly if the classifier's gradient has multiple components with the same (max) value.
- #919 Fixed `CConstraintL1` crashing when projecting sparse data using default center value (scalar 0).
- #920 Fixed inconsistent results between dense and sparse data for `CConstraintL1` projection caused by type casting.

### Removed & Deprecated (1 change)
- #922 Removed unnecessary parameter `discrete` from `COptimizerPGDLS` and `COptimizerPGDExp`.

### Documentation (2 changes)
- #100017 Improved documentation of `CAttackEvasion`, `COptimizer`, `CLineSearch`, and corresponding subclasses.
- #918 Installing the latest stable version of RobustBench instead of the master version.


## v0.14 (23/03/2021)
- #795 Added new package `adv.attacks.evasion.foolbox` with a wrapper for [Foolbox](https://foolbox.readthedocs.io/en/stable/).
- #623 `secml` is now tested for compatibility with Python 3.8.
- #861 N-Dimensional input is now accepted by `CArray`.
- #853 Added new notebook tutorial with an application on Android Malware Detection.
- #859 Add a new tutorial notebook containing example usage and attack against RobustBench models.
- #845 Static Application Security Testing (SAST) using [bandit](https://github.com/PyCQA/bandit) is now executed during testing process.

### Requirements (5 changes)
- #623 `secml` is now tested for compatibility with Python 3.8.
- #623 The following dependencies are now required: `scipy >= 1.3.2`, `scikit-learn >= 0.22`, `matplotlib >= 3`.
- #623 The `pytorch` extra component now installs: `torch >= 1.4`, `torchvision >= 0.5`.
- #623 The `cleverhans` extra component is now available on Python < 3.8 only, due to `tensorflow 1` compatibility.
- #822 Dropped official support of Python 3.5, which reached End Of Life on 13 Sep 2020. SecML may still be usable in the near future on Python 3.5 but we stopped running dedicated tests on this interpreter.

### Added (3 changes)
- #795 Added new package `adv.attacks.evasion.foolbox` with a wrapper for [Foolbox](https://foolbox.readthedocs.io/en/stable/).
- #880 Added new `shape` parameter to the following `CArray` methods: `get_data`, `tondarray`, `tocsr`, `tocoo`, `tocsc`, `todia`, `todok`, `tolil`, `tolist`. The reshaping operation is performed after casting the array to the desired output data format.
- #855 Added new ROC-related performance metrics: `CMetricFNRatFPR`, `CMetricTHatFPR`, `CMetricTPRatTH`, `CMetricFNRatTH`.

### Improved (3 changes)
- #861 N-Dimensional input is now accepted by `CArray`. If the number of dimensions of input data is higher than 2, the data is reshaped to 2 dims, and the original shape is stored in the new attribute `input_shape`.
- #910 The MNIST dataset loader `CDataLoaderMNIST` now downloads the files from our model-zoo mirror (https://gitlab.com/secml/secml-zoo/-/tree/datasets/MNIST).
- #886 Torch datasets now stored by `CDataLoaderTorchDataset` in a "pytorch" subfolder of `SECML_DS_DIR` to avoid naming collisions.

### Fixed (8 changes)
- #897 Fixed crash in `CAttackPoisoning` when `y_target != None` due to missing broadcasting to expected shape.
- #873 Use equality instead of identity to compare literals (fixing related SyntaxWarning in Python 3.8).
- #867 Now calling `StandardScaler`, `CScalerNorm`, `CScalerMinMax` arguments using keywords to fix scikit futurewarning in version 0.23 or later.
- #870 Filtering "DeprecationWarning: tostring() is deprecated. Use tobytes() instead." raised by tensorflow 1.15 if numpy 1.19 is installed.
- #868 Correctly escaping latex commands in docstrings to avoid "DeprecationWarning: invalid escape sequence \\s".
- #871 Fixed `ValueError: k exceeds matrix dimensions` not raised by scipy v1.5 if a `k` outside the array dimensions is used to extract a diagonal.
- #872 Fixed scipy 1.5 not always keeping the dtype of the original array during getitem (especially if the result is an empty array).
- #888 Filter warning raised by torchvision mnist loader first time you download.

### Removed & Deprecated (2 changes)
- #875 Removed parameter `frameon` from `CFigure.savefig` as it is deprecated in matplotlib >= 3.1.
- #875 Removed parameter `papertype` from `CFigure.savefig` as it is deprecated in matplotlib >= 3.3.

### Documentation (10 changes)
- #853 Added new notebook tutorial with an application on Android Malware Detection.
- #859 Add a new tutorial notebook containing example usage and attack against RobustBench models.
- #898 Added "Open in Colab" button to all tutorial notebooks.
- #899 Added "Edit on Gitlab" button to doc pages.
- #900 Moved notebook 11 "Evasion Attacks on ImageNet (Computer Vision)" to "Applications" section.
- #905 Changed image used by notebook 8, as the previous one is no more available.
- #903 Updated roadmap page in documentation.
- #890 Fixed multiple typos and improved language in the README.
- #878 Updated intersphinx mapping for numpy's documentation.
- #850 Fixed `MNIST` typo in notebook 10.


## v0.13 (24/07/2020)
- #814 Added new evasion attack `CAttackEvasionPGDExp`.
- #780 Added new classifier `CClassifierDNR` implementing Deep Neural Rejection (DNR). See *Sotgiu et al. “Deep neural rejection against adversarial examples”, EURASIP J. on Info. Security (2020)*.
- #47 Added new classifier `CClassifierMulticlassOVO` implementing One-vs-One multiclass classification scheme.
- #765 Extended `CModule` to support trainable modules via `fit` and `fit_forward` functions.
- #800 Security evaluation can now be run using Cleverhans attacks. The name of the parameter to check should be specified as `attack_params.<param_name>` as an input argument for the constructor of `CSecEval`.
- #839 Experimental support of Windows operating system (version 7 or later).

### Requirements (1 change)
- #768 Removed temporary pin of Pillow to v6 which used to break torch and torchvision packages.

### Added (4 changes)
- [#100007](https://gitlab.com/secml/secml/-/issues/100007) Added new experimental package `ml.scalers` with a different implementation of `ml.features.normalization` classes directly based Scikit-Learn's scalers. Included classes are: `CScalerMinMax`, `CScalerStd`, `CScalerNorm`.
- #770 Added new methods to convert a `CArray` to specific `scipy.sparse` array formats: `tocoo`, `tocsc`, `todia`, `todok`, `tolil`.
- #812 `CAttackPoisoning` now exposes: `x0`, `xc`, `yc`, `objective_function` and `objective_function_gradient`.
- #776 `n_jobs` is now a init parameter of `CModule` and subclasses and not passed via `fit` anymore.

### Improved (12 changes)
- #817 Added `CClassifierSVM` native support to OVA multiclass scheme, without replicating the kernel in each one-vs-all classifier.
- #574 Added `_clear_cache` mechanism to `CModule` and classes that require caching data in the forward pass before backward (e.g., exponential kernels do that to avoid re-computing the kernel matrix in the backward pass).
- #820 Add parallel execution of `forward` method for `CClassifierMulticlassOVA` and `CClassifierMulticlassOVO`.
- #815 Simplified `CAttack` interface (now only requires implementing `run` as required by `CSecEval`).
- #574 Modified kernel and classifier interfaces to allow their use as preprocessing modules.
- #775 Improved efficiency in gradient computation of SVMs, by back-propagating the alpha values to the kernel.
- #773 Improved efficiency in the computation of gradients of evasion attacks (`CAttackEvasionPGDLS`). Now gradient is called once rather than twice to compute the gradient of the objective function.
- #801 `CSecEval` will now check that the `param_name` input argument can be found in the attack class used in the evaluation.
- #695 `COptimizerPGD` now exits optimization if constraint radius is 0. `COptimizerPGD` , `COptimizerPGDLS` and `COptimizerPGDExp` will now raise a warning if the 0-radius constraint is defined outside the given bounds.
- #828 `CClassifierSVM` now uses `n_jobs` parameter for parallel execution of training in case of multiclass datasets.
- #767 Using `scipy.sparse` `.hstack` and `.vstack` instead of a custom implementation in `CSparse.concatenate`.
- #772 Using `scipy.sparse` `.argmin` and `.argmax` instead of a custom implementation in `CSparse.argmin` and `CSparse.argmax`.

### Changed (6 changes)
- #817 Kernel is now used as preprocess in `CClassifierSVM`.
- #817 Removed `store_dual_vars` and `kernel.setter` from `CClassifierSVM`. Now a linear SVM is trained in the primal (w,b) if `kernel=None`, otherwise it is trained in the dual (alpha and b), on the precomputed training kernel matrix.
- #765 Unified `fit` interface from `fit(ds)` to `fit(x,y)` to be consistent across normalizers and classifiers.
- #574 Removed redundant definitions of `gradient(x, w)` from `CKernelRBF`, `CKernelLaplacian`, `CKernelEuclidean`, `CClassifierDNN`, `CNormalizerUnitNorm`. The protected property `grad_requires_forward` now specifies if gradient has to compute an explicit forward pass or only propagate the input `x` through the pre-processing chain before calling `backward`.
- #823 Removed `surrogate_data` parameter from `CAttackPoisoning` and renamed it to `double_init_ds` in `CAttackEvasion` subclasses.
- #829 `CClassifierRejectThreshold` now returns wrapped classifier classes plus the reject class (-1).

### Fixed (10 changes)
- #816 Fixed stop condition of `COptimizerPGD` which was missing index `i`.
- #825 Infer the number of attacked classifier classes directly from it (instead of inferring it from surrogate data) in `CAttackEvasionPGDLS` to fix a crash when the class index of data points is greater or equal than the number of alternative data points.
- #810 Fixed `CClassifierPyTorch.backward` not working properly due to a miscalculation of the number of input features of the model when a `CNormalizeDNN` is used as preprocessor.
- #803 Fixed checks on the inner classifier in `CClassifierRejectThreshold` which can be bypassed by using the clf attribute setter, now removed.
- #818 Fixed `CCreator.set` not allowing to set writable attributes of level-0 readable-only attributes.
- #819 Fixed `CCreator.get_params` not returning level-0 not-writable attributes having one or more writable attributes.
- #785 Fixed constant override of matplotlib backend in `CFigure` on Windows systems.
- #783 Fixed `model_zoo.load_model` improperly building download urls depending on the system default url separator.
- #771 Fixed the following methods of `CSparse` to ensure they properly work independently from the sparse array format: `save`, `load`, `__pow__`, `round`, `nan_to_num`, `logical_and`, `unique`, `bincount`, `prod`, `all`, `any`, `min`, `max`.
- #769 `CArray.tocsr()` now always returns a `scipy.sparse.csr_matrix` array as expected.

### Removed & Deprecated (2 changes)
- #540 Removed `discrete` and `surrogate_classifier` parameter from `CAttack`.
- #777 Deprecated attribute `kernel` is now removed from `CClassifierSGD`, `CClassifierRidge` and `CClassifierLogistic` classifiers.

### Documentation (10 changes)
- #839 Windows is now displayed as a supported Operating System in README and setup.
- #806 Documented pytorch extra component installation requirements under Windows.
- #834 Temporarily pinned `numpydoc` to `< 1.1` to avoid compatibility issues of the newest version.
- #807 Documentation is now built using Sphinx https://readthedocs.org/ theme v0.5 or higher.
- #830 Fixed links to repository pages by adding a dash after project name.
- #758 Added a direct link to the gitlab.com repository in README.
- #788 Notebooks now include a warning about the required extra components (if any).
- #787 Fixed argmin -> argmax typo in docstring of `CClassifierRejectThreshold.predict` method.
- #789 Fixed notebook 4 not correctly generating a separate dataset for training the target classifiers.
- #791 Fixed `random_state` not set for `CClassifierDecisionTree` in notebook 4.


## v0.12 (11/03/2020)
- #726 Refactored kernel package (now `secml.ml.kernels`). Kernel classes are now inherited from `CModule`, which enables computing gradients more efficiently. This will enable us to use kernels as preprocessors in future releases.
- #755 Package `secml.ml.model_zoo` has been moved to `secml.model_zoo`.
- #721 Dictionary with model zoo definitions is now dynamically downloaded and updated from our repository located at https://gitlab.com/secml/secml-zoo. The package `model_zoo.models` containing python scripts defining models structure is now removed and the scripts will be downloaded from the same repository upon request.

### Added (7 changes)
- #660 `CClassifierPyTorch` now accepts as input a PyTorch learning rate scheduler via the `optimizer_scheduler` parameter.
- #678 Added new parameter `return_optimizer` to `CClassifierPyTorch.get_state` which allows getting the state of the classifier without including the state of the `optimizer` (and of the new `optimizer_scheduler`).
- Added `random_state` parameter to `CPSKMedians`.
- Added the parameter `minlength` to the `bincount` method of `CArray`.
- Added new `CNormalizerTFIDF` which implements a term frequency–inverse document frequency features normalizer.
- #666 Added new `utils.download_utils.dl_file_gitlab` function which allows downloading a file from a [gitlab.com](https://gitlab.com) repository, including branch and access token setting.
- #722 Added new optional parameter `headers` to `utils.download_utils.dl_file` function which allows specifying additional headers for the download request.

### Improved (8 changes)
- #664 The following `CClassifierPyTorch` parameters can now be modified after instancing the class: `optimizer`, `epochs`, `batch_size`. This will make some procedures easier, like fine-tuning a pre-trained network.
- #712 `download_utils.dl_file()` will now use the filename stored in response's headers if available. The previous behavior (get the last part of the download url) will be used as a fallback.
- #748 `CNormalizerUnitNorm` re-factored by adding gradient computation.
- #706 Rewrite `CKernelRBF` gradient when passing `w` to speed up computations by avoiding broadcasting.
- #730 `CClassifierPyTorch` has been modified to clean cached outputs and save memory when caching such data is not required.
- Internally optimized variables can be stored inside the attack class and fetched when needed.
- Accurate evaluation of objective function for some cleverhans attacks (CW, Elastic Net).
- #666 Model zoo downloader `ml.model_zoo.load_model` function will now try to download the version of a requested model corresponding to the version of secml. If not found, the latest 'master' version of the model will be downloaded instead.

### Changed (3 changes)
- #664 When passing pre-trained models to `CClassifierDNN` and subclasses the new `pretrained` parameter should now be set to `True`. Optionally, an array of classes on which the model has been pre-trained can be passed via the new `pretrained_classes` parameter. If `pretrained_classes` is left `None`, the number of classes will be inferred from the size of the last DNN layer as before.
- `CConstraintL2.project(x)` projects now `x` onto a hypersphere of radius `radius-tol`, with `tol=1e-6`. This conservative projection ensures that `x` is projected always inside the hypersphere, overcoming projection violations due to numerical rounding errors.
- `CModule.gradient` is not calling `forward` anymore, but only prepares data for `backward`. The forward step is not required, indeed, for modules that implement analytical gradients rather than autodiff.

### Fixed (10 changes)
- #677 Fixed `CClassifierPyTorch.get_state` crashing when optimizer is not defined.
- #134 Fixed passage of `n_jobs` parameter to `CDataLoaderPyTorch` in `CClassifierPyTorch` where 2 processes are being used by the loader even if `n_jobs` is set to 1. The default value for parameter `num_workers` in `CDataLoaderPyTorch` is now correctly 0.
- #749 Fixed `CArray.argmin` and `.argmax` returning float types when applied to sparse arrays of float dtype.
- Gradient is now correctly computed in `CClassifierPytorch` even if `softmax_outputs` are active.
- #707 Fixed initial value of the objective function being computed before starting point projection in `COptimizerPGDLS`.
- #667 Fixed `download_utils.dl_file()` not removing url parameters from the name of the stored file.
- #715 `download_utils.dl_file()` now correctly manage the absence of the 'content-length' header from response.
- Inverted sign of computed kernel similarity (to have a distance measure).
- #710 Random seed in `CClassifierPyTorch` is now correctly applied also when running on the CuDNN backend.
- #639: Objective function parameter (`objective_function`) in `CAttackEvasionCleverhans` is now correctly populated for `ElasticNetMethod` and `SPSA` attacks.

### Removed & Deprecated (5 changes)
- #748 `CNormalizerUnitNorm.inverse_transform` has been removed (it only worked if one inverted `x` after transforming it, but not if other transforms were applied in between).
- Removed the parameters `n_feats` and `n_classes` from the interface of `CAttackEvasionCleverhans`.
- #744 Deprecate kernel parameter from `CClassifierSGD` and `CClassifierRidge` and removed deprecated parameter `kernel='linear'` from notebook `01-Training.ipynb`.
- #643 Removed deprecated parameter `random_seed` from `CClassifierLogistic`. Use `random_state` instead.
- #643 Removed deprecated method `is_linear` from `CClassifier`, `CNormalizer`, and related subclasses.

### Documentation (5 changes)
- #756 Fixed format of output arrays reported in `CArray.__mul__` and `.__truediv__` methods.
- #681 Fixed few typos in `CExplainerIntegratedGradients`.
- #674 Added `CClassifierDNN` to the documentation.
- #711 Added a "How to cite SecML" section in README.
- #703 Updated copyright notice in README.


## v0.11.2 (07/01/2020)
- This version brings fixes for a few reported issues with `CAttack` and subclasses, along with the new Developers and Contributors guide.

### Requirements (1 change)
- #700 Temporarily pinned `Pillow` to v6 to avoid breaking `torch` and `torchvision` packages.

### Fixed (7 changes)
- #698 Fixed `CAttackEvasionCleverhans` definition of `class_type`.
- #662 The number of function and gradient evaluations made during double initialization in `CAttackEvasionPGDLS` are now correctly considered by `.f_eval` and `.grad_eval` properties.
- #699 Fixed batch processing in `CClassifierPyTorch` not working properly if the number of samples to be classified is not a multiple of `batch_size`.
- #691 Function and gradient evaluation counts in `CAttackEvasionCleverhans` returned by `.f_eval` and `.grad_eval` properties now only consider the last optimized sample, consistently with other `CAttack` subclasses.
- #701 Default value of `double_init` parameter in `CAttackEvasionPGDLS` set to True as originally intended.
- #684 The solution returned by `COptimizerPGD` is now always the best one found during the minimization process.
- #697 Fixed unittests failing under numpy v1.18 due to a change in the errors raised by `genfromtxt`.

### Documentation (2 changes)
- #671 Added Developers and Contributors guide.
- #694 Added a new notebook tutorial on advanced evasion attacks using Deep Neural Networks and ImageNet dataset.


## v0.11.1 (18/12/2019)
- Fixed compatibility issues with recently released scikit-learn v0.22 and scipy v1.4.

### Fixed (3 changes)
- #687 Fixed reshaping of sparse arrays to vector-like when using Scipy v1.4.
- #686 Replaced deprecated import of `interp` function from scipy namespace instead of numpy namespace.
- #668 Fixed unittests failing under scikit-learn v0.22 due to a change in their class output.


## v0.11 (02/12/2019)
- #653 Added new `secml.ml.model_zoo` package, which provides a zoo of pre-trained SecML models. The list of available models will be greatly expanded in the future. See https://secml.gitlab.io/secml.ml.model_zoo.html for more details.
- #629 Greatly improved the performance of the `grad_f_x` method for `CClassifier` and `CPreProcess` classes, especially when nested via `preprocess` attribute.
- #613 Support for Python 2.7 is dropped. Python version 3.5, 3.6, or 3.7 is now required.

### Requirements (2 changes)
- #633 The following dependencies are now required: `numpy >= 1.17`, `scipy >= 1.3.1`, `scikit-learn >= 0.21` `matplotlib = 3`.
- #622 Removed dependency on `six` library.

### Added (5 changes)
- #539 Added new core interface to get and set the state of an object instance: `set_state`, `get_state`, `save_state`, `load_state`. The state of an object is a simple human-readable Python dictionary object which stores the data necessary to restore an instance to a specific status. Please not that to guarantee the exact match between the original object instance and the restored one, the standard save/load interface should be used.
- #647 Added new function `core.attr_utils.get_protected` which returns a protected attribute from a class (if exists).
- #629 `CClassifier` and `CPreProcess` classes now provide a `gradient` method, which computes the gradient by doing a forward and a backward pass on the classifier or preprocessor function chain, accepting an optional pre-multiplier `w`.
- #539 Added new accessible attributes to multiple classes: `CNormalizerMinMax .m .q`; `CReducerLDA .lda`; `CClassifierKNN .tr`; `CClassifierRidge .tr`; `CClassifierSGD .tr`; `CClassifierPyTorch .trained`.
- #640 Added `random_state` parameter to `CClassifierDecisionTree`.

### Improved (6 changes)
- #631 Data objects are now stored using protocol 4 by `pickle_utils.save`. This protocol adds support for very large objects, pickling more kinds of objects, and some data format optimizations.
- #639 Objective function parameter (`objective_function`) in `CAttackEvasionCleverhans` is now correctly populated for the following attacks: `CarliniWagnerL2`, `FastGradientMethod`, `ProjectedGradientDescent`, `LBFGS`, `MomentumIterativeMethod`, `MadryEtAl`, `BasicIterativeMethod`.
- #638 The sequence of modifications to the attack point (`x_seq` parameter) is now correctly populated in `CAttackEvasionCleverhans`.
- #595 A pre-trained classifier can now be passed to `CClassifierRejectThreshold` to avoid running fit twice.
- #627 Slight improvement of `CKernel.gradient()` method performance by removing unnecessary calls.
- #630 Sparse data can now be used in `CKernelHistIntersect`.

### Changed (2 changes)
- #616 Renamed `CModelCleverhans` to `_CModelCleverhans` as this class is not supposed to be explicitly used.
- #111 Default value of the parameter `tol` changed from `-inf` to `None` in `CClassifierSGD`. This change should not alter the classifier behavior when using the default parameters.

### Fixed (8 changes)
- #611 Fixed `CDataloaderMNIST` crashing depending on the desired number of samples and digits to load.
- #652 Number of gradient computations returned by `CAttackEvasionCleverhans.grad_eval` is now accurate.
- #650 Fixed `CAttackEvasionCleverhans.f_eval` wrongly returns the number of gradient evaluations.
- #637 Fixed checks on `y_taget` in `CAttackEvasionCleverhans` which compared the 0 label to untargeted case (`y_true = None`).
- #648 Function `core.attr_utils.is_public` now correctly return False for properties.
- #649 Fixed wrong use of `core.attr_utils.is_public` in `CCreator` and `CDatasetHeader`.
- #655 Fixed `CClassifierRejectThreshold.n_classes` not taking into account the rejected class (label -1).
- #636 Fixed a `TypeError` raised by `CFigure.clabel()` when using matplotlib 3.

### Removed & Deprecated (4 changes)
- #628 Method `is_linear` of `CClassifier` and `CNormalizer` subclasses is now deprecated.
- #641 Parameter `random_seed` of `CClassifierLogistic` is now deprecated. Use `random_state` instead.
- #603 Removed deprecated class `CNormalizerMeanSTD`.
- #603 Removed deprecated parameter `batch_size` from `CKernel` and subclasses.

### Documentation (4 changes)
- #625 Reorganized notebooks tutorials into different categories: *Machine Learning*, *Adversarial Machine Learning*, and *Explainable Machine Learning*.
- #615 Added a tutorial notebook on the use of Cleverhans library wrapper.
- #607 Settings module `secml.settings` is now correctly displayed in the docs.
- #626 Added missing reference to `CPlotMetric` class in docs.


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
- #602 Renamed `CNormalizerMeanSTD` to `CNormalizerMeanStd`. The old class has been deprecated and will be removed in a future version.

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
