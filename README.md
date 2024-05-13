# SecML: Secure and Explainable Machine Learning in Python

[![Status Alpha](https://img.shields.io/badge/status-alpha-yellow.svg)](.)
[![Python 3.6 | 3.7 | 3.8](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-brightgreen.svg)](.)
[![Platform Linux | MacOS | Windows ](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey.svg)](.)
[![Apache License 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

SecML is an open-source Python library for the **security evaluation** of
Machine Learning algorithms.
It is equipped with **evasion** and **poisoning** adversarial machine learning attacks, 
and it can **wrap models and attacks** from other different frameworks.

## Table of Contents
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Tutorials](#tutorials)
* [Contributing](#contributing)
* [How to cite SecML](#how-to-cite-secml)
* [Contacts](#contacts)
* [Acknowledgements](#acknowledgements)
* [Copyright](#copyright)


## Getting Started

SecML can run with Python >= 3.6 with no additional configuration steps 
required, as all its dependencies are available as wheel packages for 
the principal macOS versions, Linux distributions and Windows.

### Installation

1. Install the latest version of ``setuptools``:

```bash
pip install -U setuptools
```

2. Install from official PyPI repository:
```bash
pip install secml
```

In all cases, the setup process will try to install the correct dependencies.
In case something goes wrong during the install process, try to install
the dependencies **first** by calling: 
```python
pip install -r requirements.txt
```


### Extra Components

SecML comes with a set of extras components that can be installed if desired.
To specify the extra components to install, add the section `[extras]` while
calling `pip install`. `extras` will be a comma-separated list of components 
you want to install. Example:
```bash
pip install secml[extra1,extra2]
```
The following extra components are available:
 - `pytorch` : Neural Networks (NNs) through [PyTorch](https://pytorch.org/) deep learning platform.  
   Installs: `torch >= 1.4`, `torchvision >= 0.5`  
   *Windows only*: the url to installation archives should be manually provided as 
   `pip install secml[pytorch] -f https://download.pytorch.org/whl/torch_stable.html`.
 - `foolbox` : Wrapper of [Foolbox](https://github.com/bethgelab/foolbox), 
   a Python toolbox to create adversarial examples that fool neural networks.   
   Installs: `foolbox >= 3.3.0`, `eagerpy >= 0.29.0`, `torch >= 1.4`, `torchvision >= 0.5`
 - `cleverhans` : Wrapper of [CleverHans](https://github.com/tensorflow/cleverhans), 
   a Python library to benchmark vulnerability of machine learning systems to adversarial examples.  
   Installs: `tensorflow >= 1.14.*, < 2`, `cleverhans < 3.1`  
   *Warning*: not available for `python >= 3.8`
 - `tf-gpu` : Shortcut for installing `TensorFlow` package with GPU support (Linux and Windows only).  
   Installs: `tensorflow-gpu >= 1.14.*, < 2`  
   *Warning*: not available for `python >= 3.8`

### Advanced features
To support additional advanced features (like the usage of GPUs) more packages can be necessary
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

## Usage
Here we show some of the key features of the SecML library.

**Wide range of supported ML algorithms.** All supervised learning algorithms
  supported by `scikit-learn` are available:
```python
# Wrapping a scikit-learn classifier
from sklearn.svm import SVC
from secml.ml.classifiers import CClassifierSkLearn
model = SVC()
secml_model = CClassifierSkLearn(model)
```
Also, SecML supports Neural Networks (NNs) through [PyTorch](https://pytorch.org/) deep learning platform:
```python
# Wrapping a Pytorch network
from torchvision.models import resnet18
from secml.ml.classifiers import CClassifierPyTorch
model = resnet18(pretrained=True)
secml_model = CClassifierPyTorch(model, input_shape=(3, 224, 224))
```

**Management of datasets.** SecML can bundle together samples and labels together in a single object:
```python
from secml.array import CArray
from secml.data import CDataset

x = CArray.randn((200, 10))
y = CArray.zeros(200)
dataset = CDataset(x, y)
```
Also, you can load famous datasets as well:
```python
from secml.data.loader import CDataLoaderMNIST
digits = (1, 5, 9)  # load subset of digits
loader = CDataLoaderMNIST()
num_samples = 200
train_set = loader.load('training', digits=digits)
test_set = loader.load('testing', digits=digits, num_samples=num_samples)
```

**Built-in attack algorithms.** Evasion and poisoning attacks based on a
  custom-developed fast solver. In addition, we provide connectors to other 
  third-party Adversarial Machine Learning libraries.
```python
from secml.adv.attacks import CAttackEvasionPGD

distance = 'l2'  # type of perturbation 'l1' or 'l2'
dmax = 2.5  # maximum perturbation
lb, ub = 0., 1.  # bounds of the attack space. None for unbounded
y_target = None  # None if untargeted, specify target label otherwise

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.5, # step size of the attack
    'max_iter': 100, # number of gradient descent steps
}

attack = CAttackEvasionPGD(classifier=secml_model,
                           distance=distance,
                           dmax=dmax,
                           solver_params=solver_params,
                           y_target=y_target)

adv_pred, scores, adv_ds, f_obj = attack.run(x, y)
```

A more detailed example covering evasion and poisoning attacks built-in in SecML can be found in [this](tutorials/06-MNIST_dataset.ipynb) notebook.

**Wrapper of other adversarial frameworks.** Attacks can also be instantiated using other framework as well.
In particular, SecML can utilizes algorithms from `foolbox` and `cleverhans`.
```python
from secml.adv.attacks import CFoolboxPGDL2
y_target = None
steps = 100
epsilon = 1.0 # maximum perturbation
attack = CFoolboxPGDL2(classifier=secml_model,
                       y_target=y_target,
                       epsilons=epsilon,
                       steps=steps)

adv_pred, scores, adv_ds, f_obj = attack.run(x, y)
```

A more detailed example covering attacks wrapped from other libraries can be found in [this](tutorials/15-Foolbox.ipynb) notebook.


**Dense/Sparse data support.** We provide full, transparent support for both
  dense (through `numpy` library) and sparse data (through `scipy` library)
  in a single data structure.
```python
from secml.array import CArray

x = CArray.zeros((4, 4))
x[0, 2] = 1
print(x)

"""
>> CArray([[0. 0. 1. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]])
"""
x = x.tosparse()
print(x) 

"""
>>  CArray((0, 2)  1.0)
"""
```

A more detailed example covering the usage of sparse data with an application in Android Malware Classification can be found in [this](tutorials/13-Android-Malware-Detection.ipynb) notebook.


**Visualize your results.** We provide a visualization and plotting framework,
  based on the widely-known library [matplotlib](https://matplotlib.org/).
```python
from secml.figure import CFigure
from secml.optim.constraints import CConstraintL2

fig = CFigure(width=5, height=5, markersize=12)

fig.subplot(1, 2, 1)

# Plot the attack objective function
fig.sp.plot_fun(attack.objective_function, 
                plot_levels=False,
                n_grid_points=200)

# Plot the decision boundaries of the classifier
fig.sp.plot_decision_regions(secml_model, 
                             plot_background=False, 
                             n_grid_points=200)

# Plot the optimization sequence
fig.sp.plot_path(attack.x_seq)

# Plot a constraint
fig.sp.plot_constraint(constraint)

fig.title("SecML example")

fig.show()
```

**Explain your results.** Explainable ML methods to interpret model decisions
  via influential features and prototypes.
```python
from src.secml.explanation import CExplainerIntegratedGradients

# Compute explanations (attributions) w.r.t. each class
attributions = CArray.empty(shape=(dataset.num_classes, x.size))
for c in dataset.classes:
    attributions_c = CExplainerIntegratedGradients(clf).explain(x, y=c)
    attributions[c, :] = attributions_c

# Visualize the explanations
fig = CFigure()

# Threshold to plot positive and negative relevance values symmetrically
threshold = max(abs(attributions.min()), abs(attributions.max()))

# Plot explanations
for c in dataset.classes:
    fig.sp.imshow(attributions[c, :].reshape((dataset.header.img_h, 
                                              dataset.header.img_w)),
                  cmap='seismic', vmin=-1 * threshold, vmax=threshold)
    fig.sp.yticks([])
    fig.sp.xticks([])
fig.show()
```

A more detailed example covering explainability techniques can be found in [this](tutorials/10-Explanation.ipynb) notebook.


**Model Zoo.** Use our pre-trained models to save time and easily replicate 
  scientific results.
```python
from secml.model_zoo import load_model
clf = load_model('mnist159-cnn')
```
  
## Tutorials
We provide tutorials that cover more advanced usages of SecML, and they can be found inside the [tutorials](tutorials) folder.

## Contributing

The contributing and developer's guide is available at: 
https://secml.readthedocs.io/en/latest/developers/

## How to cite SecML
If you use SecML in a scientific publication, please cite the following paper:

[secml: A Python Library for Secure and Explainable Machine Learning](
https://arxiv.org/abs/1912.10013), Melis *et al.*, arXiv preprint arXiv:1912.10013 (2019).

```bibtex
@article{melis2019secml,
  title={secml: A Python Library for Secure and Explainable Machine Learning},
  author={Melis, Marco and Demontis, Ambra and Pintor, Maura and Sotgiu, Angelo and Biggio, Battista},
  journal={arXiv preprint arXiv:1912.10013},
  year={2019}
}
```

## Contacts
The best way for reaching us is by opening issues. However, if you wish to contact us, you can drop an email to:
* [maura.pintor@unica.it](mailto:maura.pintor@unica.it)
* [luca.demetrio93@unica.it](mailto:luca.demetrio93@unica.it)


## Acknowledgements
SecML has been partially developed with the support of European Union’s 
[ALOHA project](https://www.aloha-h2020.eu/) Horizon 2020 Research and 
Innovation programme, grant agreement No. 780788, and Horizon Europe [ELSA – European Lighthouse on Secure and Safe AI](https://elsa-ai.eu), grant agreement No. 101070617.


## Copyright
SecML has been developed by [PRALab - Pattern Recognition and Applications lab](
https://pralab.diee.unica.it) and [Pluribus One s.r.l.](https://www.pluribus-one.it/) 
under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). All rights reserved.


