##################
Code Contributions
##################

Contribution consisting on new code the library are much welcome.
One of our maintainers will review the contributions and will help
with the needed changes before integration, if any.

Development Installation
========================

Start by creating a `fork <https://gitlab.com/secml/secml/-/forks/new>`_
of our repository. Then, set up the project locally by the following means:

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
See `how to create a merge request <https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html>`_
guide for more information.

Editable Installation
---------------------

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

Merge request checklist
=======================

Please follow this checklist before sending a new merge request:

1.  Use informative names for classes, methods, functions and variables.
2.  Make sure your code passes the existing tests.
    Remember to test both CPU and GPU (CUDA) mode, if applicable.
3.  Make sure your code is well documented and commented when possible.
    Make sure the documentation renders properly by compiling it.
4.  Add tests if you are contributing to a new feature.
5.  Make sure your code does not violate `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_
    codestyle convention.
6.  When applicable, re-use the code in the library without rewriting
    procedures that are already implemented somewhere else.
7.  (optional) Provide an example of usage in the merge request, so that
    the contribution to the library will become clear to
    the reviewers as well as other contributors.

Coding guidelines
=================

In this section, we summarize standards and conventions used in our library.

Code style
----------

We follow python `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_
convention for ensuring code readability. 4-spaces indentation should be used.

Documentation style
-------------------

We use informative docstrings for our code. Make sure your code is always
commented and documented. The docstrings should follow the
`NumPy documentation format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

To locally build the documentation, run the following command:
``tox -e docs``.

Compiled files will be available in the ``docs/build/html`` folder.

Packages
--------

Our packages are nested inside macro-categories. Every package can contain
modules, other packages or just directories for keeping everything structured
and tidy.

Modules
-------

We use separate files for each class so that they can be easily
found within the package structure. Modules can also be created to group
utility functions together.

Classes
-------

Our class names all start with ``C + <class_name>``, e.g. ``CClassifier``.
Hidden utility classes, accessible only internally from other classes,
have names starting with underscores (``_C + <class_name>``).

The packages's superclass often expose public methods that call inner
abstract methods. If you are subclassing one of these classes, take care of
reading the superclass code and check out the inner methods that you need
to implement. See: `Extending SecML <contributing.extensions.html>`_.

Tests
-----

We test our code with pervasive unit tests, build on Python's
`unittest <https://docs.python.org/3/library/unittest.html>`_ framework.
Existing unittests can be run using `tox <https://tox.readthedocs.io/>`_.

You can also contribute to writing tests for our code. We have ``tests``
subdirectories in all our packages.

The main interface from which new tests should be inherited is the
:class:`secml.testing.c_unittest.CUnitTest` class.

Tests should run smoothly and fast, performing accurate initialization and
cleanup. Implement the initialization in the ``setUp`` method,
and the single test cases in other separate methods.

New test modules should be have name starting with ``test_``.
