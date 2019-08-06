"""
.. module:: Constants
   :synopsis: Collection of mathematical constants

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import numpy as np
import math

__all__ = ['inf', 'nan', 'eps', 'e', 'pi']


"""Positive infinity."""
inf = np.inf


"""Not a number."""
nan = np.nan


"""Machine epsilon.

This is defined as the smallest number that, when
added to one, yields a result different from one.

Notes
-----
This value can be different from machine to machine,
but generally yelds approximately 1.49e-08.

Examples
--------
>>> from secml.core.constants import eps
>>> print(eps)
1.4901161193847656e-08

"""
eps = np.sqrt(np.finfo(float).eps)


"""The mathematical constant e = 2.718281..., to available precision.

Examples
--------
>>> from secml.core.constants import e
>>> print(e)
2.718281828459045

"""
e = math.e


"""The mathematical constant pi = 3.141592..., to available precision.

Examples
--------
>>> from secml.core.constants import pi
>>> pi
3.141592653589793

"""
pi = math.pi
