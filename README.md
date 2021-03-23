# SecML: A library for Secure and Explainable Machine Learning

SecML is an open-source Python library for the **security evaluation** of
Machine Learning (ML) algorithms.

It comes with a set of powerful features:
- **Wide range of supported ML algorithms.** All supervised learning algorithms
  supported by `scikit-learn` are available, as well as Neural Networks (NNs)
  through [PyTorch](https://pytorch.org/) deep learning platform.
- **Built-in attack algorithms.** Evasion and poisoning attacks based on a
  custom-developed fast solver. In addition, we provide connectors to other 
  third-party Adversarial Machine Learning libraries.
- **Dense/Sparse data support.** We provide full, transparent support for both
  dense (through `numpy` library) and sparse data (through `scipy` library)
  in a single data structure.
- **Visualize your results.** We provide a visualization and plotting framework,
  based on the widely-known library [matplotlib](https://matplotlib.org/).
- **Explain your results.** Explainable ML methods to interpret model decisions
  via influential features and prototypes.
- **Model Zoo.** Use our pre-trained models to save time and easily replicate 
  scientific results.
- **Multi-processing.** Do you want to save time further? We provide full
  compatibility with all the multi-processing features of `scikit-learn` and
  `pytorch`, along with built-in support of the [joblib](
  https://joblib.readthedocs.io/) library.
- **Extensible.** Easily create new components, like ML models or attack 
  algorithms, by extending the provided abstract interfaces.
  
### Source code hosted at: https://gitlab.com/secml/secml

**SecML is currently in development.**
If you encounter any bugs, please report them using the 
[GitLab issue tracker](https://gitlab.com/secml/secml/-/issues).
Also, have a look at our [ROADMAP](https://secml.gitlab.io/roadmap.html) 
for an overview of the future development directions.

[![Status Alpha](https://img.shields.io/badge/status-alpha-yellow.svg)](.)
[![Python 3.6 | 3.7 | 3.8](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-brightgreen.svg)](.)
[![Platform Linux | MacOS | Windows ](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey.svg)](.)
[![Apache License 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)


## Installation Guide

We recommend installing SecML in a specific environment along with its dependencies.

Common frameworks to create and manage envs are [virtualenv](https://virtualenv.pypa.io) 
and [conda](https://conda.io). Both alternatives provide convenient user 
guides on how to correctly set up the envs, so this guide will not cover 
the configuration procedure.

### Operating System requirements

SecML can run under Python >= 3.6 with no additional configuration steps 
required, as all its dependencies are available as wheel packages for 
the principal macOS versions, Linux distributions and Windows.

However, to support additional advanced features more packages can be necessary
depending on the Operating System used:

 - Linux (Ubuntu 16.04 or later or equivalent distribution)
   - `python3-tk` for running MatplotLib Tk-based backends;
   - [NVIDIA® CUDA® Toolkit](
        https://developer.nvidia.com/cuda-toolkit) for GPU support.
      
 - macOS (10.12 Sierra or later)
   - Nothing to note.
   
 - Windows (7 or later)
   - [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](
        https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads).
   - [NVIDIA® CUDA® Toolkit](
        https://developer.nvidia.com/cuda-toolkit) for GPU support.


### Installation process

Before starting the installation process try to obtain the latest version
of the `pip` manager by calling: `pip install -U pip`

The Python package `setuptools` manages the setup process.
Be sure to obtain the latest version by calling: `pip install -U setuptools`

Once the environment is set up, SecML can installed and run by multiple means:

 1. Install from official PyPI repository:
    - `pip install secml`
    
 2. Install from wheel/zip package (https://pypi.python.org/pypi/secml#files):
    - `pip install <package-file>`

In all cases, the setup process will try to install the correct dependencies.
In case something goes wrong during the install process, try to install
the dependencies **first** by calling: `pip install -r requirements.txt`

SecML should now be importable in python via: `import secml`.

To update a current installation using any of the previous methods, 
add the `-U` parameter after the `pip install` directive.
Please see our [Update Guides](https://secml.gitlab.io/update.html) for specific
upgrade instructions depending on the source and target version.


## Extra Components

SecML comes with a set of extras components that can be installed if desired.

To specify the extra components to install, add the section `[extras]` while
calling `pip install`. `extras` will be a comma-separated list of components 
you want to install. Example:
 - `pip install secml[extra1,extra2]`

All the installation procedures via `pip` described above allow definition
of the `[extras]` section.

### Available extra components
 - `pytorch` : Neural Networks (NNs) through [PyTorch](https://pytorch.org/) deep learning platform.  
   Installs: `torch >= 1.4`, `torchvision >= 0.5`  
   *Windows only*: the url to installation archives should be manually provided as 
   `pip install secml[pytorch] -f https://download.pytorch.org/whl/torch_stable.html`.
 - `foolbox` : Wrapper of [Foolbox](https://github.com/bethgelab/foolbox), 
   a Python toolbox to create adversarial examples that fool neural networks.   
   Installs: `foolbox >= 3.3.0`, `eagerpy >= 0.29.0`, `torch >= 1.4`, `torchvision >= 0.5`
 - `cleverhans` : Wrapper of [CleverHans](https://github.com/tensorflow/cleverhans), 
   a Python library to benchmark vulnerability of machine learning systems to adversarial examples.  
   Installs: `tensorflow >= 1.14.*, < 2`, `cleverhans`  
   *Warning*: not available for `python >= 3.8`
 - `tf-gpu` : Shortcut for installing `TensorFlow` package with GPU support (Linux and Windows only).  
   Installs: `tensorflow-gpu >= 1.14.*, < 2`  
   *Warning*: not available for `python >= 3.8`


## Usage Guide

SecML is based on [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), 
[scikit-learn](https://scikit-learn.org/) and [pytorch](https://pytorch.org/), 
widely-used packages for scientific computing and machine learning with Python.

As a result, most of the library interfaces should be pretty familiar 
to frequent users of those packages.

The primary data class is the `secml.array.CArray`, multi-dimensional
(currently limited to 2 dimensions) array structure which embeds both dense
and sparse data accepting as input `numpy.ndarray` and `scipy.sparse.csr_matrix`
(more sparse formats will be supported soon). This structure is the standard
input and output of all other classes in the library.

The `secml.ml` package contains all the Machine Learning algorithms and
support classes, including classifiers, loss and regularizer functions,
kernels and performance evaluation functions. Also, a zoo of pre-trained 
models is provided by the `secml.model_zoo` package.

The `secml.adv` package contains evasion and poisoning attacks based on a
custom-developed solver, along with classes to easily perform security
evaluation of Machine Learning algorithms.

The `secml.explanation` package contains different explainable 
Machine Learning methods that allow interpreting classifiers decisions 
by analyzing the relevant components such as features or training prototypes.

The `secml.figure` package contains a visualization and plotting framework
based on [matplotlib](https://matplotlib.org/).


## Developers and Contributors

The contributing and developer's guide is available at: 
https://secml.gitlab.io/developers/


## How to cite SecML
If you use SecML in a scientific publication, please cite the following paper:

[secml: A Python Library for Secure and Explainable Machine Learning](
https://arxiv.org/abs/1912.10013), Melis *et al.*, arXiv preprint arXiv:1912.10013 (2019).

BibTeX entry:

```bibtex
@article{melis2019secml,
  title={secml: A Python Library for Secure and Explainable Machine Learning},
  author={Melis, Marco and Demontis, Ambra and Pintor, Maura and Sotgiu, Angelo and Biggio, Battista},
  journal={arXiv preprint arXiv:1912.10013},
  year={2019}
}
```


## Authors
This library is maintained by 
[PRALab - Pattern Recognition and Applications Lab](https://pralab.diee.unica.it).

List of contributors:
 - Marco Melis [1]
 - Ambra Demontis [1]
 - Maura Pintor [1], [2]
 - Battista Biggio [1], [2]

[1] Department of Electrical and Electronic Engineering, University of Cagliari, Italy  
[2] Pluribus One, Italy


## Credits
- `numpy` Travis E, Oliphant. "A guide to NumPy", USA: Trelgol Publishing, 2006.
- `scipy` Travis E. Oliphant. "Python for Scientific Computing", Computing in 
  Science & Engineering, 9, 10-20, 2007.
- `scikit-learn` [Pedregosa et al., "Scikit-learn: Machine Learning in Python", 
  JMLR 12, pp. 2825-2830, 2011.](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
- `matplotlib` [J. D. Hunter, "Matplotlib: A 2D Graphics Environment", 
  Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.](
  https://doi.org/10.1109/MCSE.2007.55)
- `pytorch` Paszke, Adam, et al. "Automatic differentiation in pytorch.", NIPS-W, 2017.
- `cleverhans` [Papernot, Nicolas, et al. "Technical Report on the CleverHans v2.1.0 
  Adversarial Examples Library." arXiv preprint arXiv:1610.00768 (2018).](
  https://arxiv.org/abs/1610.00768)
- `foolbox`   [Rauber, Jonas, et. al. "Foolbox: A Python toolbox to benchmark the 
  robustness of machine learning models", arXiv preprint arXiv:1707.04131 
  (2017).](https://arxiv.org/abs/1707.04131)


## Acknowledgements
SecML has been partially developed with the support of European Union’s 
[ALOHA project](https://www.aloha-h2020.eu/) Horizon 2020 Research and 
Innovation programme, grant agreement No. 780788.


## Copyright
SecML has been developed by [PRALab - Pattern Recognition and Applications lab](
https://pralab.diee.unica.it) and [Pluribus One s.r.l.](https://www.pluribus-one.it/) 
under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). All rights reserved.
