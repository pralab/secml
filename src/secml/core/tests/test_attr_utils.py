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

        # All cases... ugly but works :D
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub')) == set(['a']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'r')) == set(['_b']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'rw')) == set(['_c']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub+r')) == set(['a', '_b']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub+rw')) == set(['a', '_c']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub+pro')) == set(['a', '_d']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'r+rw')) == set(['_b', '_c']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'r+pro')) == set(['_b', '_d']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'rw+pro')) == set(['_c', '_d']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub+r+rw')) == set(['a', '_b', '_c']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub+r+pro')) == set(['a', '_b', '_d']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub+rw+pro')) == set(['a', '_c', '_d']))
        self.assertTrue(set(attr for attr in extract_attr(f, 'pub+r+rw+pro')) == set(['a', '_b', '_c', '_d']))


if __name__ == '__main__':
    CUnitTest.main()
