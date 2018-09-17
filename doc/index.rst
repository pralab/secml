.. src documentation master file, created by
   sphinx-quickstart on Thu Nov 27 19:36:01 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: images/advlib2_logo.png
   :alt: AdversariaLib 2

=============================
AdversariaLib 2 Documentation
=============================

The library is built on multiple packages:

* :ref:`prlib`
* :ref:`advlib`

The followings are the minimal requirements:

* Python (>= 2.6 or >= 3.3),
* NumPy (>= 1.8.2),
* SciPy (>= 0.12).

.. _prlib:

Pattern Recognition and Machine Learning
========================================

The Pattern Recognition Library (prlib) implements and support a wide range of commonly used algorithms for Machine Learning 
under pattern recognition scenarios (but not limited to!). Alongwith :doc:`Classifiers <prlib.learning>`, :doc:`Estimators <prlib.learning>`, :doc:`Kernels <prlib.kernel>` and :doc:`Matchers <prlib.similarity>`, various utility functions are available, 
like :doc:`Arrays <prlib.array>` and :doc:`prlib.data` containers, :doc:`Plot <prlib.figure>` creation and more!

See :doc:`Pattern Recognition Library <prlib>` main page for more informations.

.. toctree::
   :titlesonly:

   prlib.array
   prlib.learning
   prlib.core
   prlib.data
   prlib.features
   prlib.figure
   prlib.kernel
   prlib.optimization
   prlib.parallel
   prlib.peval
   prlib.similarity
   prlib.utils

.. _advlib:

Adversarial Learning
====================

As learning algorithms typically assume data stationarity, we developed a set of functions useful to evaluate the security of machine learning (ML)-based classifiers under adversarial attacks.

See :doc:`Adversarial Learning Library <advlib>` main page for more informations.

.. toctree::
   :titlesonly:

   advlib.constraints
   advlib.evasion
   advlib.poisoning

Indices and Tables
==================

* :ref:`General Index <genindex>`
* :ref:`modindex`

