# SecML: A library for Secure and Explainable Machine Learning

SecML is an open-source Python library for the **security evaluation** of
Machine Learning (ML) algorithms.

It comes with a set of powerful features:
- **Dense/Sparse data support.** We provide full, transparent support for both
  dense (through `numpy` library) and sparse data (through `scipy` library)
  in a single data structure.
- **Wide range of supported ML algorithms.** All supervised learning algorithms
  supported by `scikit-learn` are available, as well as Neural Networks (NNs)
  through [PyTorch](https://pytorch.org/) deep learning platform _(coming soon)_.
- **Built-in attack algorithms.** Evasion and poisoning attacks based on a
  custom-developed fast solver.
- **Visualize your results.** We provide visualization and plotting framework
  based on the widely-known library [matplotlib](https://matplotlib.org/).
- **Explain your results.** Explainable ML methods to interpret model decisions
  via influential features and prototypes. _(coming soon)_  
- **Extensible.** Easily create new wrappers for ML models or attack algorithms
  extending our abstract interfaces.
- **Multi-processing.** Do you want to save time further? We provide full
  compatibility with all the multi-processing features of `scikit-learn` and
  `pytorch`, along with built-in support of the [joblib](
  https://joblib.readthedocs.io/) library.

### SecML is currently in development.
If you encounter any bug, please report them using the 
[GitLab issue tracker](https://gitlab.com/secml/secml/issues).  
Please see our [ROADMAP](https://secml.gitlab.io/roadmap.html) for an overview 
of the future development directions.

[![Status Alpha](https://img.shields.io/badge/status-alpha-yellow.svg)](.)
[![Python 2.7 | 3.5 | 3.6 | 3.7](https://img.shields.io/badge/python-2.7%20%7C%203.5%20%7C%203.6%20%7C%203.7-brightgreen.svg)](.)
[![Platform Linux | MacOS ](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey.svg)](.)
[![Apache License 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)


## Installation Guide

We recommend instaling SecML in a specific environment along with its dependencies.

Common frameworks to create and manage envs are [virtualenv](https://virtualenv.pypa.io) 
and [conda](https://conda.io). Both alternatives provide convenient user guides on 
how to properly setup the envs, so this guide will not cover the configuration 
procedure.

### Operating System requirements

SecML can run under Python 2.7 and Python >= 3.5 with no configuration steps 
required, as all its dependencies are available as wheel packages for the main 
macOS versions and Linux distributions.

However, to support additional advanced features more packages can be necessary
depending on the Operating System used:

 - Linux (Ubuntu >= 16.04 or equivalent dist):
   - `python-tk` (Python 2.7), `python3-tk` (Python >= 3.5), for running
     MatplotLib Tk-based backends;
   - NVIDIA<sup>®</sup> CUDA<sup>®</sup> Toolkit for running `tf-gpu`
     [extra component](#extra-components).
     See the [TensorFlow Guide](https://www.tensorflow.org/install/gpu).
      
 - macOS (macOS >= 10.12 Sierra)


### Installation process

Before starting the installation process try to obtain the latest version
of the `pip` manager by calling: `pip install -U pip`

The setup process is managed by the Python package `setuptools`.
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


## Extra Components

SecML comes with a set of extras components that can be installed if desired.

To specify the extra components to install, add the section `[extras]` while
calling `pip install`. `extras` will be a comma-separated list of components 
you want to install. Example:
 - `pip install secml[extra1,extra2]`

All the installation procedures via `pip` described above allow definition
of the `[extras]` section.

### Available extra components
 - None at the moment.

### _Coming soon_
 - `pytorch` : Neural Networks (NNs) through [PyTorch](https://pytorch.org/) deep learning platform.  
   Will install: `torch >= 0.4.*`, `torchvision >= 0.1.8`
 - `cleverhans` : Wrapper of [CleverHans](https://github.com/tensorflow/cleverhans), 
   a Python library to benchmark vulnerability of machine learning systems
   to adversarial examples. Will install: `tensorflow >= 1.14.*, < 2`, `cleverhans`
 - `tf-gpu` : Shortcut for installing `TensorFlow` package with GPU support.  
   Will install: `tensorflow-gpu >= 1.14.*, < 2`


## Usage Guide

SecML is based on [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), 
[scikit-learn](https://scikit-learn.org/) and [pytorch](https://pytorch.org/), 
widely-used packages for scientific computing and machine learning with Python.

As a result, most of the interfaces of the library should be pretty familiar 
to frequent users of those packages.

The primary data class is the `secml.array.CArray`, multi-dimensional
(currently limited to 2 dimensions) array structure which embeds both dense
and sparse data accepting as input `numpy.ndarray` and `scipy.sparse.csr_matrix`
(more sparse formats will be supported soon). This structure is the standard
input and output of all other classes in the library.

The `secml.ml` package contains all the Machine Learning algorithms and
support classes, including classifiers, loss and regularizer functions,
kernels and performance evaluation functions.

The `secml.adv` package contains evasion and poisoning attacks based on a
custom-developed solver, along with classes to easily perform security
evaluation of Machine Learning algorithms.

The `secml.figure` package contains a visualization and plotting framework
based on [matplotlib](https://matplotlib.org/).

_(coming soon)_ The `secml.explanation` package contains few different
explainable Machine Learning methods that allow interpreting classifiers
decisions by analyzing the relevant components such as features or training
prototypes.

_(coming soon)_ The `secml.pytorch` package contains support classes for the
[PyTorch](https://pytorch.org/) deep learning platform. This package will be
available only if the extra component `pytorch` has been specified during installation.

_(coming soon)_ The `secml.tf.clvhs` package contains support classes for the
[CleverHans](https://github.com/tensorflow/cleverhans) library for benchmarking
machine learning systems' vulnerability to adversarial examples. 
This package will be available only if the extra component `cleverhans`
has been specified during installation.


## Contributors

**Your contribution is foundamental!**

If you want to help the development of SecML, just set up the project locally
by the following means:

 1. _(devs only)_ Install from local GitLab repository:
    - Clone the project repository in a directory of your choice
    - Run installation as: `pip install .`
    
 2. _(devs only)_ Install from remote GitLab repository. In this case, given
    `{repourl}` in the format, es., `gitlab.com/secml/secml`:
    - `pip install git+ssh://git@{repourl}.git[@branch]#egg=secml`
    A specific branch to install can be specified using `[@branch]` parameter.
    If omitted, the default branch will be installed.
    
Contributions can be sent in the form of a merge request via our 
[GitLab issue tracker](https://gitlab.com/secml/secml/issues).
    
SecML can also be added as a dependency for other libraries/project.
Just add `secml` or the full repository path command 
`git+ssh://git@{repourl}.git[@branch]#egg=secml` to the `requirements.txt` file.

#### Editable Installation (development mode)

For SecML developers or users want to use the latest `dev` version
of the library, `pip` provides a convenient option which is called: **editable mode**.

By calling `pip install` with the `-e` option or `python setup.py develop`,
only a reference to the project files is "installed" in the active
environment. In this way, project files can be edited/updated and the
new versions will be automatically executed by the Python interpreter.

Two common scenarios are listed below:

 1. Editable install from a previously cloned local repository
    - Navigate to the repository directory
    - Run `python setup.py develop`
    
 2. Editable install from remote repository
    - Run `pip install -e git+ssh://git@{repourl}.git[@branch]#egg=secml`
    - Project will be cloned automatically in `<venv path>/src/secml`
    - The new repository can then be updated using standard `git` commands

Editable installs are also available while using SecML as a
dependency of other libraries/projects
(see [Installation Guide](#installation-guide) for more information).


## Authors
This library is maintained by 
[PRALab - Pattern Recognition and Applications Lab](https://pralab.diee.unica.it).

List of contributors:
 - Marco Melis (maintainer) [1]_
 - Ambra Demontis [1]_
 - Maura Pintor [1]_ , [2]_
 - Battista Biggio [1]_ , [2]_

.. [1] Department of Electrical and Electronic Engineering, University of Cagliari, Italy  
.. [2] Pluribus One, Italy


## Credits
- `numpy` Travis E, Oliphant. "A guide to NumPy", USA: Trelgol Publishing, 2006.
- `scipy` Travis E. Oliphant. "Python for Scientific Computing", Computing in 
  Science & Engineering, 9, 10-20, 2007.
- `scikit-learn` [Pedregosa et al., "Scikit-learn: Machine Learning in Python", 
  JMLR 12, pp. 2825-2830, 2011.](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
- `matplotlib` [J. D. Hunter, "Matplotlib: A 2D Graphics Environment", 
  Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.](
  https://doi.org/10.1109/MCSE.2007.55)


## Copyright
SecML has been developed by [PRALab - Pattern Recognition and Applications lab](
https://pralab.diee.unica.it) and [Pluribus One s.r.l.](https://www.pluribus-one.it/) 
under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Copyright 2019.
