from secml.testing import CUnitTest

from secml.core.attr_utils import extract_attr


class TestAttributeUtilities(CUnitTest):
    """Unit test for secml.core.attr_utils."""

    def test_extract_attr(self):
        # Toy class for testing
        class Foo:
            def __init__(self):
                self.a = 5
                self._b = 5
                self._c = 5
                self._d = 5

            @property
            def b(self):
                pass

            @property
            def c(self):
                pass

            @c.setter
            def c(self):
                pass

        f = Foo()

        self.logger.info(
            "Testing attributes extraction based on accessibility...")

        def check_attrs(code, expected):
            self.assertTrue(
                set(attr for attr in extract_attr(f, code)) == expected)

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
