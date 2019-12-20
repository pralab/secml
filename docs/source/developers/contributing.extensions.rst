###############
Extending SecML
###############

We provide details on how to implement new library modules, extending our
abstract interfaces.

Remember to check out
`latest version <https://gitlab.com/secml/secml/-/releases>`_ and
`roadmap <https://secml.gitlab.io/roadmap.html>`_ before developing new code.

Abstract Base Classes
=====================

The packages's abstract superclasses (e.g. :class:`CClassifier`) expose
public methods that call inner abstract methods. If you are creating a new
extension, inherit from one of these classes, taking care of reading the
superclass code and check out the inner methods that you need to implement.

New extensions should handle our main data type :class:`CArray`.
This class wraps the dense numpy :class:`numpy.ndarray` and the scipy sparse
:class:`scipy.sparse.csr_matrix`, so that they have the same interface
for the user.

The shape of a ``CArray`` is either a vector or a matrix (multi-dimensional
arrays will be added in a future version) of rows where each row represents
a sample.

Two ``CArray`` are needed to compose a :class:`CDataset`
that can be used to store samples (attribute X) and labels (attribute Y).

Creating new extensions
=======================

The following guides illustrate how to extend the superclasses for the
different packages of the library:

.. toctree::
   :hidden:

   contributing.extensions.cclassifier


* `CClassifier <contributing.extensions.cclassifier.html>`_ -
  classifiers including Deep Neural Networks.

The following contribution guides will be added in a future versions.

* Data processing

  - :class:`CPreprocess`
  - :class:`CKernel`

* Data

  - :class:`CDataLoader`

* Visualization

  - :class:`CPlot`

* Attacks

  - :class:`CAttack`
  - :class:`CAttackEvasion`
  - :class:`CAttackPoisoning`

* Optimization

  - :class:`COptimizer`
