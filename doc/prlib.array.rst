Array Containers
================

We support dense and sparse arrays using a main container that manage both formats transparently. As a result it is not generally  needed the use of :class:`.CDense` or :class:`.CSparse` classes explicitly.
To pass array data to built-in functions use format conversion methods such as :meth:`.CArray.toarray` or the property :attr:`.CArray.data` .

More informations on the containers for different formats can be found in :doc:`CDense <prlib.array.dense>` and :doc:`CSparse <prlib.array.sparse>` pages.

Main data container (CArray)
----------------------------

.. automodule:: prlib.array.c_array
    :members:
    :undoc-members:
    :show-inheritance:

