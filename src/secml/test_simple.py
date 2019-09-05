"""
A simple test for checking installation/configuration.
"""

from secml.array import CArray


def test_dot():
    a = CArray([1, 2, 3])
    b = CArray([10, 20, 30])
    a.dot(b)
