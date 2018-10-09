Pairwise Kernels and Metrics
============================

We use sklearn pairwise metrics for support both dense and sparse
arrays transparently.

Forse dense format only, we developed a set of optimized rountines
that dramatically increase computational performance using `Numba <http://numba.pydata.org/>`_ optimization library. The following kernels have been currently optimized:

.. toctree::

   Numba Kernels <prlib.kernel.numba_kernel>

See :doc:`Numba Kernels <prlib.kernel.numba_kernel>` for more informations.


Kernel Interface
----------------

.. automodule:: prlib.kernel.c_kernel
    :members:
    :undoc-members:
    :show-inheritance:

Linear Kernel (CKernelLinear)
-----------------------------

.. automodule:: prlib.kernel.c_kernel_linear
    :members:
    :private-members:
    :show-inheritance:

Radial Basis Function (RBF) Kernel (CKernelRBF)
-----------------------------------------------

.. automodule:: prlib.kernel.c_kernel_rbf
    :members:
    :private-members:
    :show-inheritance:

Polynomial Kernel (CKernelPoly)
-------------------------------

.. automodule:: prlib.kernel.c_kernel_poly
    :members:
    :private-members:
    :show-inheritance:

Histogram Intersection Kernel (CKernelHistIntersect)
----------------------------------------------------

.. automodule:: prlib.kernel.c_kernel_histintersect
    :members:
    :private-members:
    :show-inheritance:

