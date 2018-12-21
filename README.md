# SecML-Lib: A library for Secure Machine Learning

SecML-Lib is an open-source Python library for the **security evaluation** of Machine Learning (ML) algorithms.

It comes with a set of powerful features:
- **Dense/Sparse data support.** We provide full, transparent support for both dense (through `numpy` library) and sparse data (through `scipy` library) in a single data structure.
- **Wide range of supported ML algorithms.** All supervised learning algorithms supported by `scikit-learn` are available, as well as Neural Networks (NNs) through [PyTorch](https://pytorch.org/) deep learning platform.
- **Built-in attack algorithms.** Evasion and poisoning (*available soon*) attacks based on gradient-descent optimization.
- **Visualize your results.** We provide visualization and plotting framework based on the widely-known library [`matplotlib`](https://matplotlib.org/).
- **Extensible.** Easily create new wrappers for ML models or attack algorithms extending our abstract interfaces.
- **Multi-processing.** Do you want to save time further? We provide full compatibility with all the multi-processing features of `scikit-learn` and `pytorch`, along with built-in support of the [`joblib`](https://joblib.readthedocs.io/) library.

### SecML-Lib is currently in development. If you encounter any bug, please report them using the GitLab issues tracker.

[![Status DEV](https://img.shields.io/badge/status-dev-red.svg)]()
[![Python 2.7](https://img.shields.io/badge/python-2.7-brightgreen.svg)]()
[![Platform MacOS | Linux](https://img.shields.io/badge/platform-macos%20%7C%20linux-lightgrey.svg)]()
[![GNU GPLv3](https://img.shields.io/badge/license-GPL%20(%3E%3D%203)-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Installation Guide
As generally recommended for any Python project, SecML-Lib should be installed 
 in a specific environment along with its dependencies. Common frameworks to 
 create and manage envs are [virtualenv](https://virtualenv.pypa.io) and 
 [conda](https://conda.io). Both alternatives provide convenient user guides on 
 how to properly setup the envs, so this guide will not cover the configuration 
 procedure.

### Operating System requirements
Depending on the Operating System (OS), few libraries must be installed before SecML-Lib
and its dependencies. Most of them are provided by default on common distributions, but 
we provide a list of required package for each OS for convenience:
- Linux (Ubuntu >= 16.04):
   - `python-dev python-pip wget build-essential pkg-config gfortran libatlas-base-dev libffi-dev libssl-dev`
   - For full `matplotlib` support: `python-tk libpng-dev libgif-dev libjpeg8-dev libtiff5-dev libpng12-dev libfreetype6-dev`
- MacOS: **TODO**
- Windows: **TODO**

### Installation process

Before starting the installation process try to obtain the latest version
of the `pip` manager by calling: `pip install -U pip`

The setup process is managed by the Python package `setuptools`. Be sure
 to obtain the latest version by calling: `pip install -U setuptools`

Once the environment is set up, SecML-Lib can installed and run by
 multiple means:
 1. Install from official PyPI repository **(not yet supported)**
    - `pip install secml-lib`
 2. Install from zip/wheel package:
    - `pip install <package-file>`
 3. Install from local GitLab repository:
    - Clone the project repository in a directory of your choice
    - Run installation as: `python setup.py install`
 4. Install from remote GitLab repository. In this case, given
    `{repourl}` in the format, es., `pragit.diee.unica.it/secml/secml-lib`:
    - `pip install git+ssh://git@{repourl}.git[@branch]#egg=secml-lib`
    A specific branch to install can be specified using `[@branch]` parameter.
    If omitted, the default branch will be installed.

In all cases, the setup process will try to install the correct dependencies.
In case something goes wrong during the install process, try to install
 the dependencies **first** by calling: `pip install -r requirements.txt`

SecML-Lib should now be importable in python via: `import secml`.

To update a current installation using any of the previous methods, add the 
 `-U` parameter after the `pip install` directive.

SecML-Lib can be added as a dependency for other libraries/project.
Just add `secml-lib` (**not yet supported**) or the full repository
path command `git+ssh://git@{repourl}.git[@branch]#egg=secml-lib` to
your `requirements.txt` file.

#### Editable Installation (development mode)

For SecML-Lib developers or users want to use the latest `dev` version
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
    - Run `pip install -e git+ssh://git@{repourl}.git[@branch]#egg=secml-lib`
    - Project will be cloned automatically in `<venv path>/src/secml-lib`
    - The new repository can then be updated using standard `git` commands

Editable installs are also available while using SecML-Lib as a
dependency of other libraries/projects (see "Installation Guide"
section for more information).

## Usage Guide

SecML-Lib is based on [`numpy`](http://www.numpy.org/), [`scipy`](https://www.scipy.org/), [`scikit-learn`](https://scikit-learn.org/) and [`pytorch`](https://pytorch.org/), widely-used packages for scientific 
computing and machine learning with Python. As a result, most of the interfaces of the 
library should be pretty familiar to frequent users of those packages.

The primary data class is the `secml.array.CArray`, multi-dimensional (currently limited to 2 dimensions) array structure which embeds both dense and sparse data accepting as input `numpy.ndarray` and `scipy.sparse.csr_matrix` (more sparse formats will be supported soon). This structure is the standard input and output of all other classes in the library.

The `secml.ml` package contains all the Machine Learning algorithms and support classes, including classifiers, loss and regularizer functions, kernels and performance evaluation functions.

The `secml.adv` package contains evasion and poisoning (*available soon*) attacks based on gradient-descent optimization, along with classes to easily perform security evaluation of Machine Learning algorithms.

The `secml.figure` package contains a visualization and plotting framework based on [`matplotlib`](https://matplotlib.org/).

**[optional]** The `secml.pytorch` package contains support classes for the [PyTorch](https://pytorch.org/) deep learning platform. This package will be available only if the installation of `pytorch` library has been allowed during installation.

## Credits
SecML-Lib has been developed by [PRALab](https://pralab.diee.unica.it) - Pattern Recognition and Applications lab under [GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. Copyright 2019.

Authors (Mail to: `<name>.<surname>@diee.unica.it`):
- Marco Melis (maintainer)
- Ambra Demontis
- Battista Biggio

Contributors:
- Davide Maiorca
- Igino Corona
- Paolo Russu

