"""
A simple test for checking installation/configuration.
"""

from secml.array import CArray
from secml.figure import CFigure


def test_dot():
    a = CArray([1, 2, 3])
    b = CArray([10, 20, 30])
    return a.dot(b)


fig = CFigure()
fig.sp.plot(test_dot(), marker='o')
fig.show()
