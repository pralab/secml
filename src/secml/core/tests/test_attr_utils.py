from secml.testing import CUnitTest

from secml.core import CCreator
from secml.core.attr_utils import extract_attr


class Foo(CCreator):  # Toy class for testing
    def __init__(self):
        self.a = 1  # Public
        self._b = 2  # Read-only
        self._c = 3  # Read/Write
        self._d = 4  # Protected (no read or write)

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, val):
        self._c = val


class TestAttributeUtilities(CUnitTest):
    """Unit test for secml.core.attr_utils."""

    def setUp(self):
        self.test = Foo()

    def test_extract_attr(self):
        """Test for `extract_attr`."""

        def check_attrs(code, expected):
            self.assertTrue(
                set(attr for attr in extract_attr(self.test, code)) == expected)

        check_attrs('pub', {'a'})
        check_attrs('r', {'_b'})
        check_attrs('rw', {'_c'})
        check_attrs('pub+r', {'a', '_b'})
        check_attrs('pub+rw', {'a', '_c'})
        check_attrs('pub+pro', {'a', '_d'})
        check_attrs('r+rw', {'_b', '_c'})
        check_attrs('r+pro', {'_b', '_d'})
        check_attrs('rw+pro', {'_c', '_d'})
        check_attrs('pub+r+rw', {'a', '_b', '_c'})
        check_attrs('pub+r+pro', {'a', '_b', '_d'})
        check_attrs('pub+rw+pro', {'a', '_c', '_d'})
        check_attrs('pub+r+rw+pro', {'a', '_b', '_c', '_d'})


if __name__ == '__main__':
    CUnitTest.main()
