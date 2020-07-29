"""
A simple test for checking installation/configuration.
"""

from secml.array import CArray
from secml.figure import CFigure


def test_simple():
    """Plot the result of a dot product operation."""
    def test_dot():
        a = CArray([1, 2, 3])
        b = CArray([10, 20, 30])
        return a.dot(b)

    fig = CFigure()
    fig.sp.plot(test_dot(), marker='o')
    fig.show()


if __name__ == '__main__':
    test_simple()
