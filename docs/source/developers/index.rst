************
Contributing
************

**Your contribution is foundamental!**

If you want to help the development of SecML, just set up the project locally
by the following means:

 1. Install from local GitLab repository:

    - Clone the project repository in a directory of your choice
    - Run installation as: ``pip install .``

 2. Install from remote GitLab repository. In this case, given ``{repourl}``
    in the format, es., ``gitlab.com/secml/secml``:

    - ``pip install git+ssh://git@{repourl}.git[@branch]#egg=secml``
      A specific branch to install can be specified using ``[@branch]`` parameter.
      If omitted, the default branch will be installed.

Contributions can be sent in the form of a merge request via our
`GitLab issue tracker <https://gitlab.com/secml/secml/issues>`_.

SecML can also be added as a dependency for other libraries/project.
Just add ``secml`` or the full repository path command
``git+ssh://git@{repourl}.git[@branch]#egg=secml`` to the ``requirements.txt`` file.

Editable Installation (development mode)
----------------------------------------

For SecML developers or users want to use the latest ``dev`` version of
the library (soon available to the public), ``pip`` provides a convenient
option which is called: **editable mode**.

By calling ``pip install`` with the ``-e`` option or ``python setup.py develop``,
only a reference to the project files is "installed" in the active
environment. In this way, project files can be edited/updated and the
new versions will be automatically executed by the Python interpreter.

Two common scenarios are listed below:

 1. Editable install from a previously cloned local repository

    - Navigate to the repository directory
    - Run ``python setup.py develop``

 2. Editable install from remote repository

    - Run ``pip install -e git+ssh://git@{repourl}.git[@branch]#egg=secml``
    - Project will be cloned automatically in ``<venv path>/src/secml``
    - The new repository can then be updated using standard ``git`` commands

Editable installs are also available while using SecML as a
dependency of other libraries/projects
(see `Installation Guide <https://secml.gitlab.io/#installation-guide>`_ for more information).

Ways to contribute
==================

There are many ways to contribute to our project. The most
valuable contributions for us are the ones that extend our
library by connecting it to most-used ML libraries and
by adding state-of-the art attacks and defenses to use in
experiments. Other useful contributions are documentation and
examples of usage, which will greatly help us enlarge our user
community.

You can also help by answering questions in the issue tracker,
investigating bugs and collaborating in other merge request with
other contributors.

Another way of contributing is by sharing our work with colleagues
and people that may be interested in using it for their experiments.

Submitting a bug report or feature request
------------------------------------------

Before creating an issue we kindly ask you to read the
`documentation <https://secml.gitlab.io>`_
and to make sure your answer is not already there.

Bug report
==========

Please ensure the bug was not already reported.
If you're unable to find an open issue addressing
the problem, open a
`new one <https://gitlab.com/secml/secml/issues/new>`_.
Be sure to include
a title and clear description, as much relevant
information as possible, and a code sample or an
executable test case demonstrating the expected
behavior that is not occurring.

Feature request
===============

Suggestions and feedback are always welcome.
We ask you to open an
`issue <https://gitlab.com/secml/secml/issues/new>`_,
please provide documentation and clear instructions
on what would be the expected behavior of the new
feature. Of course, you are free to contribute
yourself (the next section will address that).

Contributing to the code
------------------------

Please stay as close as possible to the following
guidelines. This will ensure quality and easy-to-merge
requests. Your request to merge will be reviewed
by one of our maintainers and you will be asked to
reformat if needed.
You can find the instructions for sending the merge request
`here <https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html>`_.
Changes will be accepted if they add substantial
improvements to stability, functionality, testability, and
documentation of the library.

Coding guidelines
-----------------

In this section, we will summarize standards and conventions
used in our library.

Code style guide
================
We follow python `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_
convention for ensuring code readability.
If you see code not compliant to this standard,
please do not open an issue for this reason.
Issues related to cosmetic changes will be closed.
If you are contributing to the code, please refer
to these conventions before committing your changes.
Do not perform refactoring or code reformat in
already-existing modules: this would cause us some
trouble in comparing the differences with git.

Documentation style guide
=========================

We use informative docstrings for our code. Make sure
your code is always commented and documented.
The docstrings follow the NumPy documentation format.

Here is the list of things regarding the documentation
that you can contribute to:

*   You can contribute to creating documentation
    for our code.
*   You can contribute to the examples in our library,
    creating a Jupyter notebook well commented and with
    the discussion of the results. We save our notebooks
    with the output so the user won’t need to run them
    for seeing the output.

TODO: guide to build the documentation

Package conventions
===================

TODO

Modules conventions
===================

TODO

Classes conventions
===================

TODO

Tests conventions
=================

TODO

Merge request checklist
=======================

Before you ask us to revise the code and merge the code
into our library, we ask you to pass all the steps
in this checklist. After that, you can send the merge request
to us and we will evaluate ourselves. We will refuse
merge requests not compliant with the following checklist,
but passing this test is only the first step. We will still
need to evaluate the code and ensure to benefit from it.

1.  Give informative names to everything. This means not
    only to give useful names to classes, method and python-related
    things, but also to the merge request itself.
2.  Make sure your code passes the tests. Remember to test both
    CPU and CUDA mode, if applicable.
3.  Make sure your code is well documented and that the
    processes inside are commented. Make sure the documentation
    renders properly by compiling it with Sphinx. Delete commented
    lines of code.
4.  Add tests if you are contributing for a new feature. Make sure
    the new feature is tested correctly and follows the expected
    behavior correctly.
5.  Make sure your code does not violate PEP-8.
    Please avoid reformatting parts of the file that
    your pull request doesn’t change, as it distracts from code review.
6.  When applicable, re-use the code in the library without rewriting
    parts that are already implemented somewhere else.
7.  (optional) Provide an example of usage in the merge request, so that
    the contribution to the library will become clear to
    the reviewers as well as other contributors.


Creating a SecML backend
------------------------


A **backend** is an interface that links third-party
libraries or stand-alone code to the SecML library.
Since there is a great number of frameworks around
the web, we cannot provide connectors to all libraries
on our own. This is why we ask our community to implement
their own modules and share them with us. We provide
this guide for the implementation of new modules,
along with examples of implementation and hints on how
to define the required modules.

SecML already contains some library connectors and
backend implementations, remember to check out
`last version <https://gitlab.com/secml/secml/-/releases>`_ and
`roadmap <https://secml.gitlab.io/roadmap.html>`_ before diving into code.

Unified backend interface
=========================

In order to use our powerful APIs, developers will
have to create **converters** to handle our
**custom data type** for python arrays, the
`CArray <https://secml.gitlab.io/secml.array.html#module-secml.array.c_array>`_,
and implement the interfaces defined in metaclasses
such as the
`CClassifier <https://secml.gitlab.io/secml.ml.classifiers.html#module-secml.ml.classifiers.c_classifier>`_.

The CArray class wraps the dense Numpy array and
the sparse `csr_matrix` so that they have the
same interface for the user.

The shape of a CArray is either a vector or a
matrix of rows where each row represents a sample.

Two CArray can be composed in a
`CDataset <https://secml.gitlab.io/secml.data.html#secml.data.c_dataset.CDataset>`_,
that can be used to store samples
(attribute X) and labels (attribute Y).


Steps for creating a new backend
================================

In this section, we list all methods to implement
for minimal support of a new backend module.

We will list several use cases, so don’t be
scared if they seem too many.

Focus on your use case, then give a read to
the methods’ description before writing the code.
This will help you design the classes and avoid mistakes.

Implementing a Classifier
=========================

SecML defines a
`unified classifier interface <https://secml.gitlab.io/
secml.ml.classifiers.html#secml.ml.classifiers.c_classifier
.CClassifier>`_ for enforcing the base structure for all
classifiers. All new classifiers, except for DNNs (the next
section will discuss this case), which have a more
specific interface, must inherit from the CClassifier
class. The class CClassifier requires the developer
to implement three private methods in order to function.

CClassifier
===========

Here is the list of methods to implement for creating
a new classifier (not DNN):

-   `_forward`: performs a forward pass of the input x.
    It should return the output of the decision function
    of the classifier.

-   `_backward`: this method returns the gradient
    of the decision function output with respect to data.
    It takes a CArray `w` as input, which pre-multiplies
    the gradient as in standard reverse-mode autodiff.

-   `_fit`: trains the One-Vs-All classifier.
    Takes as input a CDataset.

Implementing a backend for DNN
==============================

The backend for DNN ([CClassifierDNN](-))
is based on the CClassifier class as well
but adds more methods specific to DNNs and
their frameworks.

You can see how to use the `CClassifierDNN`
class in our implemented `PyTorch backend <https://secml.gitlab.io/
secml.ml.classifiers.html#module-secml.ml
.classifiers.pytorch.c_classifier_pytorch>`_.

CClassifierDNN
==============

Here is the list of methods to implement for
creating a new DNN classifier:

-   _forward: performs a forward pass of the
    input x. It is slightly different from
    the `_forward` method of the CClassifier,
    as it returns the output of the layer of the
    DNN specified in the attribute `_out_layer`.
    If `_out_layer` is None, the last layer output
    is returned (applies the softmax if
    `softmax_outputs` is True).

-   `_backward`: returns the gradient of the
    output of the DNN layer specified in
    `_out_layer`, with respect to the input data.

-   `_fit`: trains the One-Vs-All classifier.
    Takes as input a CDataset.

-   `layers` (property): returns a list of
    tuples containing the layers of the model,
    each tuple is structured as `(layer_name, layer)`.

-   `layer_shapes` (property): returns the
    output shape of each layer (as a dictionary
    with layer names as keys).

-   `_to_tensor`: converts a CArray into the
    tensor data type of the backend framework.

-   `_from_tensor`: converts a backend tensor
    data type to a CArray

-   `save_model`: saves the model weight and
    parameters into a gz archive. If possible,
    it should allow model restoring as a
    checkpoint - the user should be able to continue
    training of the restored model.

-   `load_model`: restores the model. If possible,
    it restores also the optimization parameters
    as the user may need to continue training.

It may be necessary to implement a custom data loader
for the specific backend. The data loader should take
as input a CDataset from SecML and load the data for
the backend. This is necessary because the inputs to
the network may have their own shapes, whereas the
CArray treats each sample as a row vector. We suggest
to add the `input_shape` as an input parameter of
the wrapper and handle the conversion inside.

More advanced implementations (not available yet)
=================================================

The following contribution guides will be updated in future versions.

*   Data processing

    -   `CPreprocess`

    -   `CKernel`

*   Data

    -   `CDataLoader`

*   Visualization

    -   `CPlot`

